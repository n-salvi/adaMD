from scipy.optimize import least_squares, leastsq, curve_fit, nnls
import numpy as np

def matA(tau_scale, x):
    c = np.array(x).reshape((1,len(x)))
    mat_x = np.repeat(c, len(tau_scale), axis=0)

    c = np.array(tau_scale).reshape((1,len(tau_scale)))
    mat_tau = np.repeat(c, len(x), axis=0).T

    exps = np.divide(mat_x,-mat_tau)
    return np.exp(exps)

def non_zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def exp_decay(x, a, tau):
    return a*np.exp(-np.array(x)/tau)

def exp_decay_1(x, tau):
    return np.exp(-np.array(x)/tau)

def exp_fit(x, y):
    popt, pcov = curve_fit(exp_decay, x, y, p0=(y[0], x[-1]/5.), bounds=(0, [2*y[0], x[-1]*2]))
    simulated = exp_decay(x, *popt)
    return popt, pcov, simulated

def exp_fit_1(x, y):
    popt, pcov = curve_fit(exp_decay_1, x, y, p0=x[-1]/5., bounds=(0, x[-1]*2))
    simulated = exp_decay_1(x, *popt)
    return popt, pcov, simulated

def smooth_and_fit(values, tstep=1., lp_threshold=0.15, lp_threshold_2=0.05, mintau=2., numtaus=8192):

    # tau_scale = np.logspace(np.log10(mintau), np.log10(maxtau), numtaus)

    x = [tstep*n for n in range(len(values))]

    test = np.where(values < lp_threshold)[0]
    if len(test)>0:
        lp = np.where(values < lp_threshold)[0][0]
    else:
        lp = int(len(values)/2)

    test = np.where(values < lp_threshold_2)[0]
    if len(test)>0:
        lp2 = np.where(values < lp_threshold_2)[0][0]
    else:
        lp2 = len(values)-1

    #this is the smoothing method of Lindorff Larsen JACS 2012
    tosmooth_y = values[lp:lp2]
    tosmooth_x = x[lp:lp2]

    popt, pcov = curve_fit(exp_decay, tosmooth_x, tosmooth_y, p0=(tosmooth_y[0], tosmooth_x[-1]/5.), bounds=(0, [2*tosmooth_y[0], lp2*tstep]))

    smooth = exp_decay(x, *popt)
    for timepoint in range(lp, lp2):
        ratio = -(timepoint-lp)/(lp2-lp)+1.
        values[timepoint] = (1.-ratio)*smooth[timepoint]+ratio*values[timepoint]
    for timepoint in range(lp2, len(values)):
        values[timepoint] = smooth[timepoint]

    tau_scale = np.logspace(np.log10(mintau), np.log10(popt[1]), numtaus)

    matA1 = matA(tau_scale, x)
    A_final, rnorm = nnls(matA1.transpose(), values)

    temp = non_zero_runs(A_final)

    tau_estimates = []
    amp_estimates = []
    for dec in temp:
        these_amps = A_final[dec[0]:dec[1]]
        these_taus = tau_scale[dec[0]:dec[1]]
        amp_estimates.append(np.sum(these_amps))
        tau_estimates.append(np.average(these_taus, weights=these_amps))

    sim_mat = [np.multiply(A_final[m], vec) for m, vec in enumerate(matA1)]
    simulated = np.sum(sim_mat, axis=0)

    return tau_estimates, amp_estimates, simulated, x

def TR(values, tstep=1., mintau=2., maxtau=50000, numtaus=8192, S2=False):

    tau_scale = np.logspace(np.log10(mintau), np.log10(maxtau), numtaus)

    if S2==True:
        tau_scale = np.append(tau_scale, 1e12)

    x = [tstep*n for n in range(len(values))]

    matA1 = matA(tau_scale, x)
    A_final, rnorm = nnls(matA1.transpose(), values)

    temp = non_zero_runs(A_final)

    tau_estimates = []
    amp_estimates = []
    for dec in temp:
        these_amps = A_final[dec[0]:dec[1]]
        these_taus = tau_scale[dec[0]:dec[1]]
        amp_estimates.append(np.sum(these_amps))
        tau_estimates.append(np.average(these_taus, weights=these_amps))

    sim_mat = [np.multiply(A_final[m], vec) for m, vec in enumerate(matA1)]
    simulated = np.sum(sim_mat, axis=0)

    return tau_estimates, amp_estimates, simulated, x
