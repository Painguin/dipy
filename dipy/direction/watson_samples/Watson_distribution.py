# functions to generate dispersion
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

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
for kappa in [16., 32., 64., 128.]: # Don't forget float form
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

    np.savetxt('watson_Kappa%1.1f_straight.txt' % kappa,s_watson)
