import os
import numpy as np

def check_path(outdir):
    if not outdir.endswith('/'):
        outdir = outdir+'/'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

import datetime

def get_today():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def rotation_matrix(theta, u):
    N = len(u)
    theta = theta*np.pi/180. #in radians
    IM = np.identity(N)*np.cos(theta)
    CM = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])*np.sin(theta)
    TM = np.array([[u[0]**2, u[0]*u[1], u[0]*u[2]], [u[0]*u[1], u[1]**2, u[1]*u[2]], [u[0]*u[2], u[1]*u[1], u[2]**2]])*(1.-np.cos(theta))
    return IM + CM + TM

def stddih(angles):
    angles=np.array(angles)
    sines = np.sin(angles*np.pi/180.)
    cosines = np.cos(angles*np.pi/180.)
    S = np.sum(sines)
    C = np.sum(cosines)
    N = len(angles)
    return np.sqrt(-2.*np.log(np.sqrt(S**2+C**2)/N))*180./np.pi

import numpy.random as npr

def bootstrap(data, num_samples=1000, statistic=np.mean, alpha=0.05):
    """Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic."""
    n = len(data)
    data=np.array(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha/2.0)*num_samples)],
            stat[int((1-alpha/2.0)*num_samples)])

def bootstrap_array(array, num_samples=10, statistic=np.mean, alpha=0.05):
    array=np.array(array)
    lenrow = len(array[0])
    return np.array([bootstrap(array[:, n], num_samples=num_samples, statistic=statistic, alpha=alpha) for n in range(lenrow)])

def RMSD(A, B):
    A = np.array(A)
    B = np.array(B)
    return np.sqrt(np.mean((A-B)**2))
