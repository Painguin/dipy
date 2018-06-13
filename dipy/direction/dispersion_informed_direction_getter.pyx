cimport numpy as np
import numpy as np
import random
import os

from dipy.tracking.local.direction_getter cimport DirectionGetter

cdef class DispersionInformedPeakDirectionGetter(DirectionGetter):

    cdef:
        double peak_threshold
        double cos_similarity
        int nbr_peaks
        double[:, :, :, :] data
        np.ndarray peaks
        np.ndarray kappa_params
        dict samples
        bint multi_param

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
        kappa: ndarray
            The dispersion parameter
        """
        if data.shape[-1] % 3 != 0:
            raise ValueError("data should be a 4d array of N peaks, with the "
                             "last dimension of size N*3")

        self.nbr_peaks = data.shape[-1]/3
        self.peaks = np.zeros((self.nbr_peaks, 3), dtype=float)
        for i in range(self.nbr_peaks):
            norm= np.linalg.norm(data[:,:,:,i*3:(i+1)*3], axis=3)
            norm[norm == 0] = 1
            for j in range(3):
                data[:,:,:,i*3+j] =data[:,:,:,i*3+j] / norm

        self.data = np.asarray(data,  dtype=float)
        self.peak_threshold = peak_threshold
        self.cos_similarity = np.cos(np.deg2rad(max_angle))

        def round_and_clip(k):
            if k < 0.0:
                return 0.0
            if k > 128.0:
                return 128.0
            return round(k)

        if not os.path.exists('watson_samples'):
            os.mkdir('watson_samples')

        self.kappa_params = np.vectorize(round_and_clip)(np.asarray(kappa_params, dtype=float))
        unique_params = set(np.unique(self.kappa_params))

        self.samples = {}

        if 0.0 in unique_params:
            self.samples[0] = np.array([[0.0, 1.0, 0.0]])
            unique_params.remove(0.0)

        if len(kappa_params.shape) > 3:
            self.multi_param = True
        else:
            self.multi_param = False

        for k in unique_params:
            if not os.path.exists('watson_samples/watson_Kappa%1.1f_straight.txt' % k):
                _generate_watson(k)
            with open('watson_samples/watson_Kappa%1.1f_straight.txt' % k) as f:
                self.samples[k] = np.array([line.split() for line in f.read().splitlines()], dtype=float)

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

        prev = direction[0], direction[1], direction[2]

        index = _closest_peak(self.peaks, direction, self.cos_similarity)
        if index < 0:
            return 1

        # select one vector randomly from the watson samples
        kappa = self.kappa_params[p[0],p[1],p[2]]
        if self.multi_param:
            kappa = kappa[index]
        s = random.SystemRandom().choice(self.samples[kappa])

        # sampled vector coordinates
        s_x, s_y, s_z = s

        # peak coordinates
        peak = np.array([direction[0], direction[1], direction[2]])
        p_x, p_y, p_z = peak

        # compute the cosines and sines of the angles of both vectors in spherical
        # coordinates using trirgonometry
        p_cos_phi, p_sin_phi, p_cos_theta, p_sin_theta = _cos_and_sines(p_x,p_y,p_z)
        s_cos_phi, s_sin_phi, s_cos_theta, s_sin_theta = _cos_and_sines(s_x,s_y,s_z)

        # incline the peak by the sample phi angle
        # (sin(a + c)sinb, cos(a + c), sin(a + c)cosb) =
        # ((cosa*sinc + cosc*sina)sinb, cosc*cosa - sinc*sina, (cosa*sinc + cosc*sina)cosb)
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
                          (1 - s_cos_theta)*np.outer(peak,peak)

        dispersed_peak = np.dot(rotation_matrix, inclined_p)

        # only update the direction with the dispersion if we are still within the angle tolerance
        if np.dot(prev, dispersed_peak) > self.cos_similarity:
            direction[0], direction[1], direction[2] = dispersed_peak

        return 0

def _cos_and_sines(x, y, z):
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

def _generate_watson(kappa):
    kappa = float(kappa)
    def cart2sph(x,y,z):
        azimuth = np.arctan2(y,x)
        elevation = np.arctan2(z,np.sqrt(x**2 + y**2))
        r = np.sqrt(x**2 + y**2 + z**2)
        return azimuth, elevation, r

    def sph2cart(azimuth,elevation,r):
        x = r * np.cos(elevation) * np.cos(azimuth)
        y = r * np.cos(elevation) * np.sin(azimuth)
        z = r * np.sin(elevation)
        return x, y, z

    def Rz(alpha):
        return np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])

    def Ry(beta):
        return np.array([[np.cos(beta),0,np.sin(beta)],[0,1,0],[-np.sin(beta),0,np.cos(beta)]])


    # Sample from Watson distribution - straight bundle

    mu = np.array([0.0,1.0,0.0])/np.linalg.norm([0.0,1.0,0.0]) # straight
    mu /= np.linalg.norm(mu)

    n = 100000

    # Using "Random sampling from the watson distribution" by Kim-Hung Li and Carl Ka-Fai Wong : http://www.tandfonline.com.sci-hub.io/doi/pdf/10.1080/03610919308813139?needAccess=true, implementation inspired from https://github.com/libDirectional/libDirectional/blob/master/lib/distributions/Hypersphere/WatsonDistribution.m
    s_watson = np.empty((3,n))
    s_watson[:] = np.NAN
    phi = np.empty(n)
    phi[:] = np.NAN
    theta = np.empty(n)
    theta[:] = np.NAN
    rho=4*kappa/(2*kappa+3+np.sqrt((2*kappa+3)**2 - 16*kappa))
    r=(3*rho/(2*kappa))**3 * np.exp(-3+2*kappa/rho)
    i = 0
    while i < n:
        U = np.random.uniform(0,1,(n-i,3))
        S = U[:,0]**2 / (1-rho*(1-U[:,0]**2))
        V = r*U[:,1]**2 / (1-rho*S)**3
        valid = (V<=np.exp(2*kappa*S))
        if valid.sum == 0:
            continue
        U = U[valid,:]
        S = S[valid]
        thetasNew = np.arccos(np.sqrt(S))
        U2ltHalf = (U[:,2]<0.5)
        thetasNew = np.pi*U2ltHalf+(-1)**(U2ltHalf) * thetasNew
        phisNew = 4*np.pi*U[:,2]-U2ltHalf*2*np.pi
        theta[i:i+U.shape[0]] = thetasNew
        phi[i:i+U.shape[0]] = phisNew
        i=i+U.shape[0]
    muphi,mutheta,_ = cart2sph(mu[0],mu[1],mu[2])
    s_watson[0,:],s_watson[1,:],s_watson[2,:] = sph2cart(phi,-theta+np.pi/2,1)
    s_watson=Rz(muphi-np.pi).dot(Ry(mutheta+np.pi/2)).dot(s_watson).T
    unit_y = np.array([0.0,1.0,0.0])
    s_watson = np.array([-x if np.dot(x,unit_y) < 0 else x for x in s_watson])

    np.savetxt('watson_samples/watson_Kappa%1.1f_straight.txt' % kappa,s_watson)

cdef int _closest_peak(np.ndarray[np.float_t, ndim=2] peak_dirs,
                      double* direction, double cos_similarity):
    """Update direction with the closest direction from peak_dirs.
    (this method is an exact copy of the one in dipy/direction/closest_peak_direction_getter
    but it returns the index of the selected peak instead of 0 and -1 in case of failure

    All directions should be unit vectors. Antipodal symmetry is assumed, ie
    direction x is the same as -x.

    Parameters
    ----------
    peak_dirs : array (N, 3)
        N unit vectors.
    direction : array (3,) or None
        Previous direction. The new direction is saved here.
    cos_similarity : float
        `cos(max_angle)` where `max_angle` is the maximum allowed angle between
        prev_step and the returned direction.

    Returns
    -------
    the index of the selected peak : if ``direction`` is updated
    -1 : if no new direction is found
    """

    cdef:
        size_t _len=len(peak_dirs)
        size_t i
        int closest_peak_i=-1
        double _dot
        double closest_peak_dot=0

    for i in range(_len):
        _dot = (peak_dirs[i,0] * direction[0]
                + peak_dirs[i,1] * direction[1]
                + peak_dirs[i,2] * direction[2])

        if np.abs(_dot) > np.abs(closest_peak_dot):
            closest_peak_dot = _dot
            closest_peak_i = i

    if closest_peak_dot >= cos_similarity and closest_peak_i >= 0:
        direction[0] = peak_dirs[closest_peak_i, 0]
        direction[1] = peak_dirs[closest_peak_i, 1]
        direction[2] = peak_dirs[closest_peak_i, 2]
    elif closest_peak_dot <= -cos_similarity and closest_peak_i >= 0:
        direction[0] = -peak_dirs[closest_peak_i, 0]
        direction[1] = -peak_dirs[closest_peak_i, 1]
        direction[2] = -peak_dirs[closest_peak_i, 2]
    return closest_peak_i