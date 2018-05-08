cimport numpy as np
import numpy as np
import random

from dipy.direction.closest_peak_direction_getter cimport closest_peak
from dipy.tracking.local.direction_getter cimport DirectionGetter
#from dipy.tracking.local.interpolation import trilinear_interpolate4d_c

cdef class DispersionInformedPeakDirectionGetter(DirectionGetter):

    cdef:
        double peak_threshold
        double cos_similarity
        int nbr_peaks
        double[:, :, :, :] data
        np.ndarray peaks
        np.ndarray kappa_params
        dict samples

    def __init__(self, data, max_angle, peak_threshold, kappa_params, **kwargs):
        """Create a Dispersion Informed DirectionGetter

        Parameters
        ----------
        data : ndarray, float, (..., N*3)
            Peaks data with N peaks per voxel.
            last dimention format: [Px1,Py1,Pz1, ..., PxN,PyN,PzN]
        max_angle : float (0, 90)
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters.
        peak_threshold : float
            Threshold for peak norm.
        kappa: integer
            The dispersion parameter
        """
        if data.shape[-1] % 3 != 0:
            raise ValueError("data should be a 4d array of N peaks, with the "
                             "last dimension of size N*3")

        self.nbr_peaks = data.shape[-1]/3
        self.peaks = np.zeros((self.nbr_peaks, 3), dtype=float)
        for i in range(self.nbr_peaks):
            norm= np.linalg.norm(data[:,:,:,i*3:(i+1)*3], axis=3)
            for j in range(3):
                data[:,:,:,i*3+j] =data[:,:,:,i*3+j] / norm

        self.data = np.asarray(data,  dtype=float)
        self.peak_threshold = peak_threshold
        self.cos_similarity = np.cos(np.deg2rad(max_angle))

        unique_params = set(np.unique(kappa_params))
        if len(unique_params.intersection({0.0,16.0,32.0,64.0,128.0})) > len(unique_params):
            raise ValueError("the kappa parameters array contains an invalid value"
                             "the values can only be {0,16,32,64,128}")

        self.kappa_params = np.asarray(kappa_params, dtype=int)

        self.samples = {}
        if 0 in unique_params:
            self.samples[0] = np.array([[0.0, 1.0, 0.0]])
        if 16 in unique_params:
            f = open('watson_samples/watson_Kappa16.0_straight.txt')
            self.samples[16] = np.array([line.split() for line in f.read().splitlines()], dtype=float)
            f.close()
        if 32 in unique_params:
            f = open('watson_samples/watson_Kappa32.0_straight.txt')
            self.samples[32] = np.array([line.split() for line in f.read().splitlines()], dtype=float)
            f.close()
        if 64 in unique_params:
            f = open('watson_samples/watson_Kappa64.0_straight.txt')
            self.samples[64] = np.array([line.split() for line in f.read().splitlines()], dtype=float)
            f.close()
        if 128 in unique_params:
            f = open('watson_samples/watson_Kappa128.0_straight.txt')
            self.samples[128] = np.array([line.split() for line in f.read().splitlines()], dtype=float)
            f.close()

        print "init", self.cos_similarity, max_angle

    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(self,
                                                           double[::1] point):
        """Returns best directions at seed location to start tracking.

        Parameters
        ----------
        point : ndarray, shape (3,)
            The point in an image at which to lookup tracking directions.

        Returns
        -------
        directions : ndarray, shape (N, 3)
            Possible tracking directions from point. ``N`` may be 0, all
            directions should be unique.

        """
        cdef:
            int p[3]

        for i in range(3):
            p[i] = int(point[i] + 0.5)

        for i in range(3):
            if p[i] < -.5 or p[i] >= (self.data.shape[i] - .5):
                return None
        for i in range(self.nbr_peaks):
            self.peaks[i, :] = self.data[p[0], p[1], p[2], i*3:(i+1)*3]

        return self.peaks

    cdef int get_direction_c(self, double* point, double* direction):
        """

        Returns
        -------
        status : int
            Returns 0 if `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            int p[3]
            double d[3]

        for i in range(3):
            p[i] = int(point[i] + 0.5)

        for i in range(3):
            if p[i] < -.5 or p[i] >= (self.data.shape[i] - .5):
                return 1
        for i in range(self.nbr_peaks):
            self.peaks[i, :] = self.data[p[0], p[1], p[2], i*3:(i+1)*3]

        if closest_peak(self.peaks, direction, self.cos_similarity):
            return 1

        # select one vector randomly from the watson samples
        s = random.SystemRandom().choice(self.samples[self.kappa_params[p[0], p[1], p[2]]])

        # sampled vector coordinates
        s_x, s_y, s_z = s

        # peak coordinates
        p = np.array([direction[0], direction[1], direction[2]])
        p_x, p_y, p_z = p

        ''' 
            compute the cosines and sines of the angles of both vectors in spherical
            coordinates using trigonometry
        '''
        p_cos_phi, p_sin_phi, p_cos_theta, p_sin_theta = self.__cos_and_sines(p_x,p_y,p_z)
        s_cos_phi, s_sin_phi, s_cos_theta, s_sin_theta = self.__cos_and_sines(s_x,s_y,s_z)

        '''
            incline the peak by the sample phi angle
            (sin(a + c)sinb, cos(a + c), sin(a + c)cosb) =
            ((cosa*sinc + cosc*sina)sinb, cosc*cosa - sinc*sina, (cosa*sinc + cosc*sina)cosb)
        '''
        sin_phi_sum = p_cos_phi*s_sin_phi + s_cos_phi*p_sin_phi
        cos_phi_sum = p_cos_phi*s_cos_phi - p_sin_phi*s_sin_phi
        inclined_p = np.array([sin_phi_sum*p_sin_theta,
                               cos_phi_sum,
                               sin_phi_sum*p_cos_theta])

        # rotate the new inclined peak around the original peak
        cross_prod_matrix = np.array([[0.0 , -p_z, p_y ],
                                      [p_z , 0.0 , -p_x],
                                      [-p_y, p_x , 0.0 ]])

        rotation_matrix = s_cos_theta*np.eye(3) +\
                          s_sin_theta*cross_prod_matrix +\
                          (1 - s_cos_theta)*np.outer(p,p)

        dispersed_peak = np.dot(rotation_matrix, inclined_p)

        direction[0], direction[1], direction[2] = dispersed_peak

        return 0

    def __cos_and_sines(self, x, y, z):
        xz_norm = np.sqrt(x**2 + z**2)
        cos_phi = y
        sin_phi = xz_norm

        if xz_norm == 0.0:
            if y <= 0.0:
                cos_theta = -1.0
            else:
                cos_theta = 1.0
            sin_theta = 0.0
        else:
            cos_theta = z/xz_norm
            sin_theta = x/xz_norm

        return cos_phi, sin_phi, cos_theta, sin_theta
