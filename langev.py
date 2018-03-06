import numpy as np
from scipy.linalg import expm

def udamp(t,m,g,k,T,x0,v0): #t-array of timepoints, m-mass, g-\gamma, k-spring, x0/v0-init conds
    sig = 2*T*g
    ini = np.array([x0,v0])
    xout = np.zeros((t.size,2))
    M = np.array([[0,-1],[g*1./m,k*1./m]])
    xh = xout.copy()
    xp = xh.copy()
    dt = t[1]-t[0]
    r = np.zeros((t.size,2))
    r[:,1] = np.random.normal(0.,np.sqrt(sig),t.size)
    for i in range(xout[:,1].size):
        xh[i,:] = expm((-1)*M*t[i]).dot(ini)
        taus = np.linspace(t[0],t[i],i+1)
        xp[i,:] = dt*1./2*expm((-1)*M*t[i]).dot(r[i,:])+dt*1./2*r[0,:]
        j=1
        for tau in taus[1:-1]:
            xp[i,:] = xp[i,:] + dt*1./2*expm((-1)*M*(t[i]-tau)).dot(r[j,:])
            j = j+1

    xout = xh+xp
    return xout


