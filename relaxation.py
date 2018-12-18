import pytraj as pt
import numpy as np
from pytraj import vector as va
from numpy.linalg import matrix_power
import mdtraj as md
from mdtraj.testing import get_fn

def unit_vector(vector):
    # Returns the unit vector of the vector.
    return np.array(vector) / np.linalg.norm(vector)

def phi(v):
   #Returns the phi tensor::
    v_u = unit_vector(v)
    return [[a*b for a in v_u] for b in v_u]

def S2(vectors):
    phis = np.array([phi(v) for v in vectors])
    #ave_phi = np.mean(phi, axis=0)
    ave_phi = np.array([[np.mean([p[i][j] for p in phis]) for i in [0, 1, 2]] for j in [0, 1, 2]])
    return 1.5*np.trace(matrix_power(ave_phi, 2))-0.5*np.trace(ave_phi)**2


class nh_acf:

    def __init__(self, top, trajs, rotfit=False, mask='@CA', stride=1, n_blocks=1, bframes=50000, skip=25000):
        #read sequence
        if trajs[0][-3:] == 'dcd':
            traj = pt.iterload(trajs[0], top)
        else:
            #filename, top_name = get_fn(trajs[0]), get_fn(top)
            m_traj = md.load(trajs[0], top=top)
            traj = pt.Trajectory(xyz=m_traj.xyz.astype('f8'), top=top)
        self.sequence = {res.index+1:res.name for res in traj.top.residues}
        self.top = top
        self.trajs = trajs
        n_frames = len(traj)

        self.n_indices = pt.select_atoms('@N & !(:1) & !(:PRO)', traj.top)
        self.h_indices = self.n_indices + 1
        self.nh_pairs = np.array(list(zip(self.n_indices, self.h_indices)))

        res_list = sorted([k for k, val in self.sequence.items() if val != 'PRO' and k != 1])

        self.acf=dict()
        self.S2 = dict()

        if n_blocks==1:
            for tt, t in enumerate(self.trajs):
                if t[-3:] == 'dcd':
                    traj = pt.load(t, top, stride=stride)
                else:
                    #filename, top_name = get_fn(t), get_fn(top)
                    m_traj = md.load(t, top=top)
                    traj = pt.Trajectory(xyz=m_traj.xyz.astype('f8'), top=top)
                if rotfit == True:
                    #print('RMSD fitting on '+mask)
                    _ = traj.superpose(ref=-1, mask='@CA')
                data_vec = va.vector_mask(traj, self.nh_pairs, dtype='ndarray')
                self.acf[tt] = {res_list[n]:pt.timecorr(vals, vals, order=2, tstep=1, tcorr=len(vals), norm=False, dtype='ndarray')
                    for n, vals in enumerate(data_vec)}
                self.S2[tt] = {res_list[n]:S2(vals) for n, vals in enumerate(data_vec)}
        else:
            index = 0
            bcount = 0
            for tt, t in enumerate(self.trajs):
                print(tt+1, '/', len(self.trajs))
                while bcount < n_blocks:
                    first_frame = bcount*skip
                    last_frame = first_frame+bframes
                    if t[-3:] == 'dcd':
                        traj = pt.load(t, top, frame_indices=slice(first_frame, last_frame, stride))
                    else:
                        #filename, top_name = get_fn(t), get_fn(top)
                        m_traj = md.load(t, top=top)
                        m_traj = m_traj[first_frame:last_frame:stride]
                        traj = pt.Trajectory(xyz=m_traj.xyz.astype('f8'), top=top)
                    if rotfit == True:
                            _ = traj.superpose(ref=-1, mask='@CA')
                    data_vec = va.vector_mask(traj, self.nh_pairs, dtype='ndarray')
                    self.acf[index] = {res_list[n]:pt.timecorr(vals, vals, order=2, tstep=1, tcorr=len(vals), norm=False, dtype='ndarray')
                        for n, vals in enumerate(data_vec)}
                    self.S2[index] = {res_list[n]:S2(vals) for n, vals in enumerate(data_vec)}
                    bcount += 1
                    index += 1
                bcount=0


from scipy.constants import mu_0, h, value, pi


class NMR_relaxation_rates:

    def __init__(self, amps, taus, B0_MHz, theta_deg=22):

        g15N = -2.7126 * 1e7
        g1H = value("proton gyromag. ratio")
        rNH = 1.02 * 1e-10
        dCSA = -170.0 * 1e-6

        T = B0_MHz/value("proton gyromag. ratio over 2 pi")
        omegaN = g15N*T
        omegaH = g1H*T
        omega_diff = omegaH - omegaN
        omega_sum = omegaH + omegaN
        taus = np.multiply(taus, 1e-12)
        J = self.define_J(amps, taus)
        c = dCSA*omegaN/np.sqrt(3.0)
        d = mu_0*h*g1H*g15N/(8*(pi**2)*(rNH**3))

        theta = np.deg2rad(theta_deg)

        self.R1 = ((d**2)/4.0)*(6.0*J(omega_sum)+J(omega_diff)+3.0*J(omegaN))+J(omegaN)*(c**2)
        self.R2 = ((d**2)/8.0)*(6.0*J(omega_sum)+6.0*J(omegaH)+J(omega_diff)+3.0*J(omegaN)+4.0*J(0))+((c**2)/6)*(3.0*J(omegaN)+4.0*J(0))
        self.NOE = 1.0 + (d**2)/(4.0*self.R1)*(g1H/g15N)*(6.0*J(omega_sum)-J(omega_diff))
        self.etaXY = -(np.sqrt(3.0)/6.0)*d*c*self.P2(np.cos(theta))*(3.0*J(omegaN)+4.0*J(0))
        self.etaZ = -np.sqrt(3.0)*d*c*self.P2(np.cos(theta))*J(omegaN)

        self.rates = {
        'R1': self.R1,
        'R2': self.R2,
        'NOE': self.NOE,
        'etaXY': self.etaXY,
        'etaZ': self.etaZ
        }


    def define_J(self, amps, taus):
        def J(omega):
            all_terms = [0.4*amp*taus[n]/(1.0+((omega*taus[n])**2)) for n, amp in enumerate(amps)]
            return np.sum(all_terms)
        return J

    def P2(self, x):
        return (3.0*x*x-1.0)/2.0
