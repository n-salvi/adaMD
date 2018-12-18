from scipy.special import jv, legendre, sph_harm, jacobi
from scipy.misc import factorial, comb
from numpy import floor, sqrt, sin, cos, exp, power
from math import pi
from adaMD.relaxation import unit_vector, S2
from adaMD.general import stddih
from pytraj import vector as va
import scipy.fftpack as sc

def mycorrelate(v):
    ##### probably broken use mycorrelate2 instead
    print('Warning: probably broken, use mycorrelate2 instead')
    acf = np.correlate(v, v, mode='full')
    acf = np.real(acf[int(np.floor(acf.size/2)):])
    return acf/max(acf)

def mycorrelate2(v, norm=True):
    #A = np.fft.fft(v)
        A = sc.fft(v)
        B = A*np.conjugate(A)
        #C = np.real(np.fft.ifft(B))
        C = np.real(sc.ifft(B))
        if norm==True:
            C /= C[0]
        return C[0:int(np.floor(len(C)/2))]
        #return C

def mycrosscorrelate2(v1, v2, norm=True):
    #A1 = np.fft.fft(v1)
    #A2 = np.fft.fft(v2)
    A1 = sc.fft(v1)
    A2 = sc.fft(v2)
    B = A1*np.conjugate(A2)
    #C = np.real(np.fft.ifft(B))
    C = np.real(sc.ifft(B))
    if norm == True:
        C /= C[0]
    return C[0:int(np.floor(len(C)/2))]

def direct_correlate(v, stride=1):
    N = len(v)
    Ncalc = int(len(v)/2)+1
    acf = np.real([np.mean([v[n]*np.conjugate(v[n+nc]) for n in range(N) if n+nc<N]) for nc in range(0, Ncalc, stride)])
    return np.array(acf)/max(acf)


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= np.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    """
    sina = np.sin(angle)
    cosa = np.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,         -direction[2],  direction[1]],
                      [ direction[2], 0.0,          -direction[0]],
                      [-direction[1], direction[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rot_matrices(angles, directions):
    cosines = np.cos(angles)
    sines = np.sin(angles)
    directions /= np.sqrt((directions ** 2).sum(-1))[..., np.newaxis]

    ux = directions[:, 0]
    uy = directions[:, 1]
    uz = directions[:, 2]

    R00 = cosines + ux**2*(1-cosines)
    R01 = ux*uy*(1-cosines)-uz*sines
    R02 = ux*uz*(1-cosines)+uy*sines
    R10 = uy*ux*(1-cosines)+uz*sines
    R11 = cosines + uy**2*(1-cosines)
    R12 = uy*uz*(1-cosines)-ux*sines
    R20 = uz*ux*(1-cosines)-uy*sines
    R21 = uz*uy*(1-cosines)+ux*sines
    R22 = cosines + uz**2*(1-cosines)

    return np.array([[[v, R01[n], R02[n]],[R10[n], R11[n], R12[n]],[R20[n], R21[n], R22[n]]] for n, v in enumerate(R00)])

def wignerd(j,m,n=0,approx_lim=10):
    """
    Wigner "small d" matrix. (Euler z-y-z convention)
    example:
    j = 2
    m = 1
    n = 0
    beta = linspace(0,pi,100)
    wd210 = wignerd(j,m,n)(beta)

    some conditions have to be met:
    j >= 0
    -j <= m <= j
    -j <= n <= j

    The approx_lim determines at what point
    bessel functions are used. Default is when:
    j > m+10
    and
    j > n+10

    for integer l and n=0, we can use the spherical harmonics. If in
    addition m=0, we can use the ordinary legendre polynomials.
    """

    if (j < 0) or (abs(m) > j) or (abs(n) > j):
        raise ValueError("wignerd(j = {0}, m = {1}, n = {2}) value error.".format(j,m,n)         + "Valid range for parameters: j>=0, -j<=m,n<=j.")

    if (j > (m + approx_lim)) and (j > (n + approx_lim)):
        #print ‘bessel (approximation)’
        return lambda beta: jv(m-n, j*beta)

    if (floor(j) == j) and (n == 0):
        if m == 0:
            #print ‘legendre (exact)’
            return lambda beta: legendre(j)(cos(beta))
        elif False:
            #print ‘spherical harmonics (exact)’
            a = sqrt(4.*pi / (2.*j + 1.))
            return lambda beta: a * conjugate(sph_harm(m,j,beta,0.))

    jmn_terms = {
        j+n : (m-n,m-n),
        j-n : (n-m,0.),
        j+m : (n-m,0.),
        j-m : (m-n,m-n),
        }

    k = min(jmn_terms)
    a, lmb = jmn_terms[k]

    b = 2.*j - 2.*k - a

    if (a < 0) or (b < 0):
        raise ValueError("wignerd(j = {0}, m = {1}, n = {2}) value error.".format(j,m,n)         + "Encountered negative values in (a,b) = ({0},{1})".format(a,b))

    coeff = power(-1.,lmb) * sqrt(comb(2.*j-k,k+a)) * (1./sqrt(comb(k+b,b)))

    #print ‘jacobi (exact)’
    return lambda beta: coeff* power(sin(0.5*beta),a)         * power(cos(0.5*beta),b)         * jacobi(k,a,b)(cos(beta))

def Dwigner(k, l, a, b, g):
    wd = wignerd(2,k,l)(b)
    return np.exp(-1j*k*a)*wd*np.exp(-1j*l*g)


import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import numpy as np
import pytraj as pt
from math import atan2

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = atan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = atan2(y,x)                           # phi
    return r, elev, az

class armada:

    def __init__(self, top, trajs, stride):
        #read sequence
        traj = pt.iterload(trajs[0], top)
        self.sequence = {res.index+1:res.name for res in traj.top.residues}
        self.top = top
        self.trajs = trajs
        self.stride = stride

    def iterload(self, index):
        return pt.iterload(self.trajs[index], self.top, stride=self.stride)

    def angles2vectors(self, traj, masks=[['@N', '@CA'], ['@N', '@HN']], residues=[3, 3]):
        indices = [[pt.select_atoms(traj.top, ':'+str(r)+m)[0] for m in masks[n]] for n, r in enumerate(residues)]
        data_vec = va.vector_mask(traj, indices, dtype='ndarray')
        data_vec[0] /= np.sqrt((data_vec[0] ** 2).sum(-1))[..., np.newaxis]
        data_vec[1] /= np.sqrt((data_vec[1] ** 2).sum(-1))[..., np.newaxis]
        cosines = np.sum(data_vec[0]*data_vec[1], axis=1)
        return np.arccos(cosines)

    def get_dihedrals(self):
        self.phi = dict()
        self.psi = dict()
        for tt, t in enumerate(self.trajs):
            traj = self.iterload(tt)
            temp = pt.multidihedral(traj, 'phi psi')
            phi_keys = {k:int(k[4:]) for k in temp.keys() if 'phi' in k}
            psi_keys = {k:int(k[4:]) for k in temp.keys() if 'psi' in k}
            self.phi[tt] = {val:temp[k].values for k, val in phi_keys.items()}
            self.psi[tt] = {val:temp[k].values for k, val in psi_keys.items()}

    #def cal_dihedrals

    def get_rotation_matrices(self, traj, s='NH'):
        n_frames = len(traj)
        if s=='NH':
            n_indices = pt.select_atoms('@N & !(:1) & !(:PRO)', traj.top)
            h_indices = n_indices + 1
            pairs = np.array(list(zip(n_indices, h_indices)))
            res_list = sorted([k for k, val in self.sequence.items() if val != 'PRO' and k != 1])
        if s=='CA':
            res_list = sorted([k for k, val in self.sequence.items() if k != 1])
            ca_indices = pt.select_atoms('@CA', traj.top)
            pairs = np.array(list(zip(ca_indices, ca_indices[1:])))

        vecs = va.vector_mask(traj, pairs, dtype='ndarray')

        angle = dict()
        rot_axis = dict()
        rot_mat = dict()

        for n, vals in enumerate(vecs):
            v1 = vals[:-1]
            v2 = vals[1:]
            v1 /= np.sqrt((v1 ** 2).sum(-1))[..., np.newaxis]
            v2 /= np.sqrt((v2 ** 2).sum(-1))[..., np.newaxis]
            cosines = np.sum(v1*v2, axis=1)
            angle[res_list[n]] = np.arccos(cosines)

            temp = np.cross(v2, v1)
            rot_axis[res_list[n]] = temp/np.sqrt((temp ** 2).sum(-1))[..., np.newaxis]

            rot_mat[res_list[n]] = rot_matrices(angle[res_list[n]], rot_axis[res_list[n]])

        return angle, rot_axis, rot_mat

    def get_vectors(self, traj, s='NH'):
        n_frames = len(traj)
        if s=='NH':
            n_indices = pt.select_atoms('@N & !(:1) & !(:PRO)', traj.top)
            h_indices = n_indices + 1
            pairs = np.array(list(zip(n_indices, h_indices)))
            res_list = sorted([k for k, val in self.sequence.items() if val != 'PRO' and k != 1])
        if s=='psi':
            ca_indices = pt.select_atoms('@CA', traj.top)
            c_indices = pt.select_atoms('@C', traj.top)
            pairs = np.array(list(zip(ca_indices, c_indices)))
            res_list = sorted([k for k, val in self.sequence.items()])[:-1]
        if s=='phi':
            ca_indices = pt.select_atoms('@CA', traj.top)
            n_indices = pt.select_atoms('@N & !(:1)', traj.top)
            pairs = np.array(list(zip(n_indices, ca_indices)))
            res_list = sorted([k for k, val in self.sequence.items() if k != 1])
        if s=='CA':
            res_list = sorted([k for k, val in self.sequence.items() if k != 1])
            ca_indices = pt.select_atoms('@CA', traj.top)
            pairs = np.array(list(zip(ca_indices, ca_indices[1:])))

        temp = va.vector_mask(traj, pairs, dtype='ndarray')

        vecs = {r:temp[k]/np.sqrt((temp[k] ** 2).sum(-1))[..., np.newaxis] for k, r in enumerate(res_list)}

        return vecs


class gaf:

    def __init__(self, top, trajs):
        #read sequence
        traj = pt.iterload(trajs[0], top)
        self.sequence = {res.index+1:res.name for res in traj.top.residues}
        self.top = top
        self.trajs = trajs


    def getdihedrals(self, s='phi psi', stride=2):
        self.dih = dict()

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            #read dihedral angles
            self.dih[tt] = pt.multidihedral(traj, s)

    def corrdih(self):
        corrs = dict()
        for tt, multidih in self.dih.items():
            corrs[tt] = dict()
            for k in multidih.keys():
                values = multidih[k]
                temp = np.array([v-values[0] for v in values])
                temp = np.cos(temp*np.pi/180.)
                corrs[tt][k] = mycorrelate2(temp)
        return corrs

    def ramasplit(self):

        res_list = sorted([a for a in self.sequence.keys()])
        phi_labs = [el for el in res_list[1:]]
        psi_labs = [el for el in res_list[:-1]]

        self.classdih = dict()
        temp_to_ave = dict()
        confs = ['aL', 'aR', 'betas', 'ND']

        for tt in self.dih.keys():
            self.classdih[tt] = {angle:[] for angle in self.dih[tt].keys()}
            temp_to_ave[tt] = {angle:{c:[] for c in confs} for angle in self.dih[tt].keys()}

        for tt, dihedrals in self.dih.items():
            for res in res_list:
                if res not in phi_labs:
                    label = 'psi:'+str(res)
                    self.classdih[tt][label] = ['ND' for value in dihedrals[label]]
                    temp_to_ave[tt][label]['ND'] = dihedrals[label]
                if res not in psi_labs:
                    label = 'phi:'+str(res)
                    self.classdih[tt][label] = ['ND' for value in dihedrals[label]]
                    temp_to_ave[tt][label]['ND'] = dihedrals[label]
                if res in psi_labs and res in phi_labs:
                    label_phi = 'phi:'+str(res)
                    label_psi = 'psi:'+str(res)
                    these_phi = dihedrals[label_phi]
                    these_psi = dihedrals[label_psi]
                    for phi, psi in zip(these_phi, these_psi):
                        if phi>0:
                            self.classdih[tt][label_phi].append('aL')
                            self.classdih[tt][label_psi].append('aL')
                            temp_to_ave[tt][label_phi]['aL'].append(phi)
                            temp_to_ave[tt][label_psi]['aL'].append(psi)
                        if phi < 0 and psi > -130 and psi < 50:
                            self.classdih[tt][label_phi].append('aR')
                            self.classdih[tt][label_psi].append('aR')
                            temp_to_ave[tt][label_phi]['aR'].append(phi)
                            temp_to_ave[tt][label_psi]['aR'].append(psi)
                        if phi < 0 and (psi < -130 or psi > 50):
                            self.classdih[tt][label_phi].append('betas')
                            self.classdih[tt][label_psi].append('betas')
                            temp_to_ave[tt][label_phi]['betas'].append(phi)
                            temp_to_ave[tt][label_psi]['betas'].append(psi)

        self.ave_dih = {tt:{lab:{c:np.nanmean(values) for c, values in lvalues.items()}
                        for lab, lvalues in tvalues.items()} for tt, tvalues in temp_to_ave.items()}

    def omegasplit(self):

        res_list = sorted([a for a in self.sequence.keys()])
        omega_labs = [el for el in res_list[1:]]

        temp_to_ave = dict()
        confs = ['0', 'pi']

        for tt in self.dih.keys():
            temp_to_ave[tt] = {angle:{c:[] for c in confs} for angle in self.dih[tt].keys() if 'omega:' in angle}

        for tt, dihedrals in self.dih.items():
            for res in omega_labs:
                label = 'omega:'+str(res)
                centered = [value if value>90 else value+360 for value in dihedrals[label]]
                for value in centered:
                    if value > 90 and value < 270:
                        self.classdih[tt][label].append('pi')
                        temp_to_ave[tt][label]['pi'].append(value)
                    else:
                        self.classdih[tt][label].append('0')
                        temp_to_ave[tt][label]['0'].append(value)
                self.ave_dih[tt][label] = {c:np.nanmean(values) for c, values in temp_to_ave[tt][label].items()}

    def gaf2Dcorr(self, dihs=['phi:', 'psi:', 'omega:'], deltas=[0, -1, 0], leftb=[-180, -180, 90]):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        if not hasattr(self, 'gafcorrs'):
            self.gafcorrs=dict()
        if not hasattr(self, 'vardih'):
            self.vardih=dict()
        if not hasattr(self, 'rdih'):
            self.rdih=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, dihedrals in self.dih.items():
            if tt not in self.gafcorrs.keys():
                self.gafcorrs[tt]=dict()
            if tt not in self.vardih.keys():
                self.vardih[tt]=dict()
            if tt not in self.rdih.keys():
                self.rdih[tt]=dict()
            for res in res_list:
                if res not in self.gafcorrs[tt].keys():
                    self.gafcorrs[tt][res]=dict()
                if res not in self.vardih[tt].keys():
                    self.vardih[tt][res]=dict()
                if res not in self.rdih[tt].keys():
                    self.rdih[tt][res]=dict()
                dsum = ''
                for d in dihs:
                    dsum=dsum+d
                self.gafcorrs[tt][res][dsum]=dict()
                labels=[el+str(res+deltas[n]) for n, el in enumerate(dihs)]
                if len(dihs)==2:
                    angles = [dihedrals[l] for l in labels if l in dihedrals.keys()]
                    angles1 = [[val if val > leftb[n] else val+360 for val in series] for n, series in enumerate(angles)]
                else:
                    #this is a 2D GAF! if more than two rotations are present,
                    #we assume the first N to be co-axial
                    temp1 = [dihedrals[l] for l in labels[:-1] if l in dihedrals.keys()]
                    temp11 = [[val if val > leftb[n] else val+360 for val in series] for n, series in enumerate(temp1)]
                    temp1 = np.sum(temp1, axis=0)
                    temp11 = np.sum(temp11, axis=0)
                    temp2 = dihedrals[labels[-1]]
                    temp22 = [val if val > leftb[-1] else val+360 for val in temp2]
                    angles = [temp1, temp2]
                    angles1 = [temp11, temp22]

                self.vardih[tt][res][dsum] = np.array([stddih(a)**2 for a in angles1])
                temp = np.mean(np.array(angles1[0])*np.array(angles1[1])*(np.pi/180.)**2)-np.mean(np.array(angles1[0])*(np.pi/180.))*np.mean(np.array(angles1[1])*(np.pi/180.))
                self.rdih[tt][res][dsum] = temp/np.sqrt(self.vardih[tt][res][dsum][0]*self.vardih[tt][res][dsum][1])

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            exp1 = 1j*k*np.array(angles[0])*np.pi/180.+1j*l*np.array(angles[1])*np.pi/180.
                            exp2 = 1j*kp*np.array(angles[0])*np.pi/180.+1j*l*np.array(angles[1])*np.pi/180.
                            pow1 = np.exp(exp1)
                            pow2 = np.exp(exp2)
                            self.gafcorrs[tt][res][dsum][str(k)+str(kp)+str(l)] = mycrosscorrelate2(pow1, pow2)

    def gaf2DS2(self, dihs=['phi:', 'psi:', 'omega:'], deltas=[0, -1, 0], leftb=[-180, -180, 90]):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        if not hasattr(self, 'gafS2'):
            self.gafS2=dict()
        if not hasattr(self, 'vardih'):
            self.vardih=dict()
        if not hasattr(self, 'rdih'):
            self.rdih=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, dihedrals in self.dih.items():
            if tt not in self.gafS2.keys():
                self.gafS2[tt]=dict()
            if tt not in self.vardih.keys():
                self.vardih[tt]=dict()
            if tt not in self.rdih.keys():
                self.rdih[tt]=dict()
            for res in res_list:
                if res not in self.gafS2[tt].keys():
                    self.gafS2[tt][res]=dict()
                if res not in self.vardih[tt].keys():
                    self.vardih[tt][res]=dict()
                if res not in self.rdih[tt].keys():
                    self.rdih[tt][res]=dict()
                dsum = ''
                for d in dihs:
                    dsum=dsum+d
                self.gafS2[tt][res][dsum]=dict()
                labels=[el+str(res+deltas[n]) for n, el in enumerate(dihs)]
                if len(dihs)==2:
                    angles = [dihedrals[l] for l in labels if l in dihedrals.keys()]
                    angles1 = [[val if val > leftb[n] else val+360 for val in series] for n, series in enumerate(angles)]
                else:
                    #this is a 2D GAF! if more than two rotations are present,
                    #we assume the first N to be co-axial
                    temp1 = [dihedrals[l] for l in labels[:-1] if l in dihedrals.keys()]
                    temp11 = [[val if val > leftb[n] else val+360 for val in series] for n, series in enumerate(temp1)]
                    temp1 = np.sum(temp1, axis=0)
                    temp11 = np.sum(temp11, axis=0)
                    temp2 = dihedrals[labels[-1]]
                    temp22 = [val if val > leftb[-1] else val+360 for val in temp2]
                    angles = [temp1, temp2]
                    angles1 = [temp11, temp22]

                self.vardih[tt][res][dsum] = np.array([stddih(a)**2 for a in angles1])
                temp = np.mean(np.array(angles1[0])*np.array(angles1[1])*(np.pi/180.)**2)-np.mean(np.array(angles1[0])*(np.pi/180.))*np.mean(np.array(angles1[1])*(np.pi/180.))
                self.rdih[tt][res][dsum] = temp/np.sqrt(self.vardih[tt][res][dsum][0]*self.vardih[tt][res][dsum][1])

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            exp1 = 1j*k*np.array(angles[0])*np.pi/180.+1j*l*np.array(angles[1])*np.pi/180.
                            exp2 = 1j*kp*np.array(angles[0])*np.pi/180.+1j*l*np.array(angles[1])*np.pi/180.
                            pow1 = np.exp(exp1)
                            pow2 = np.exp(exp2)
                            self.gafS2[tt][res][dsum][str(k)+str(kp)+str(l)] = np.real(np.mean(pow1)*np.conjugate(np.mean(pow2)))

    def gaf1Dcorr(self, dihs='omega:', delta=0, leftb=90):
        """
        See Bremi Bruschweiler Ernst JACS 1997. and Bremi Bruschweiler JACS 1997
        """
        if not hasattr(self, 'gafcorrs'):
            self.gafcorrs=dict()
        if not hasattr(self, 'vardih'):
            self.vardih=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, dihedrals in self.dih.items():
            if tt not in self.gafcorrs.keys():
                self.gafcorrs[tt]=dict()
            if tt not in self.vardih.keys():
                self.vardih[tt]=dict()
            for res in res_list:
                if res not in self.gafcorrs[tt].keys():
                    self.gafcorrs[tt][res]=dict()
                if res not in self.vardih[tt].keys():
                    self.vardih[tt][res]=dict()
                self.gafcorrs[tt][res][dihs]=dict()
                lab=dihs+str(res+delta)
                angles = dihedrals[lab]

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                            exp1 = -1j*k*np.array(angles)*np.pi/180.
                            exp2 = -1j*kp*np.array(angles)*np.pi/180.
                            pow1 = np.exp(exp1)
                            pow2 = np.exp(exp2)
                            self.gafcorrs[tt][res][dihs][str(k)+str(kp)] = mycrosscorrelate2(pow1, pow2)

                angles = [val if val > leftb else val+360 for val in angles]
                self.vardih[tt][res][dihs] = stddih(angles)**2

    def gaf1DS2(self, dihs='omega:', delta=0, leftb=90):
        """
        See Bremi Bruschweiler Ernst JACS 1997. and Bremi Bruschweiler JACS 1997
        """
        if not hasattr(self, 'gafS2'):
            self.gafS2=dict()
        if not hasattr(self, 'vardih'):
            self.vardih=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, dihedrals in self.dih.items():
            if tt not in self.gafS2.keys():
                self.gafS2[tt]=dict()
            if tt not in self.vardih.keys():
                self.vardih[tt]=dict()
            for res in res_list:
                if res not in self.gafS2[tt].keys():
                    self.gafS2[tt][res]=dict()
                if res not in self.vardih[tt].keys():
                    self.vardih[tt][res]=dict()
                self.gafS2[tt][res][dihs]=dict()
                lab=dihs+str(res+delta)
                angles = dihedrals[lab]

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            exp1 = -1j*k*np.array(angles)*np.pi/180.
                            exp2 = -1j*kp*np.array(angles)*np.pi/180.
                            pow1 = np.exp(exp1)
                            pow2 = np.exp(exp2)
                            self.gafS2[tt][res][dihs][str(k)+str(kp)] = np.real(np.mean(pow1)*np.conjugate(np.mean(pow2)))

                angles = [val if val > leftb else val+360 for val in angles]
                self.vardih[tt][res][dihs] = stddih(angles)**2

    def simplegafcorr(self, dihs=['phi:', 'psi:'], deltas=[0, -1], usedirect=False, stride=2):
        """
        here by 'simple' we mean that we calculate only the terms in which k=k'=l, i.e.
        we assume that the two axis are collinear (good approximation for phi/psi rotations,
        beta=176.1 degrees). See Bremi Bruschweiler Ernst JACS 1997.
        """
        self.gafcorrs=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, dihedrals in self.dih.items():
            self.gafcorrs[tt]=dict()
            for res in res_list:
                self.gafcorrs[tt][res]=dict()
                labels=[el+str(res+deltas[n]) for n, el in enumerate(dihs)]
                angles = [dihedrals[l] for l in labels if l in dihedrals.keys()]
                for k in range(-2, 3):
                    if 'exponents' in locals():
                        del exponents
                    for values in angles:
                        if 'exponents' in locals():
                            exponents += np.array(values-values[0])
                        else:
                            exponents = np.array(values-values[0])
                    powers = np.exp(1j*k*exponents*np.pi/180.)
                    if usedirect==True:
                        self.gafcorrs[tt][res][k] = direct_correlate(powers, stride=stride)
                    else:
                        self.gafcorrs[tt][res][k] = mycorrelate2(powers)

    def simplejumpcorr(self, dihs=['phi:', 'psi:'], deltas=[0, -1], usedirect=False):
        """
        here by 'simple' we mean that we calculate only the terms in which k=k'=l, i.e.
        we assume that the two axis are collinear (good approximation for phi/psi rotations,
        beta=176.1 degrees). See Bremi Bruschweiler Ernst JACS 1997.
        """
        self.jumpcorrs=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, classified in self.classdih.items():
            self.jumpcorrs[tt]=dict()
            for res in res_list:
                self.jumpcorrs[tt][res]=dict()
                labels=[el+str(res+deltas[n]) for n, el in enumerate(dihs)]
                angles = [classified[l] for l in labels if l in classified.keys()]
                for k in range(-2, 3):
                    if 'exponents' in locals():
                        del exponents
                    for l, values in enumerate(angles):
                        ave_angles = [self.ave_dih[tt][labels[l]][c] for c in values]
                        if 'exponents' in locals():
                            exponents += np.array(ave_angles-ave_angles[0])
                        else:
                            exponents = np.array(ave_angles-ave_angles[0])
                    powers = np.exp(1j*k*exponents*np.pi/180.)
                    if usedirect==True:
                        self.jumpcorrs[tt][res][k] = direct_correlate(powers, stride=2)
                    else:
                        self.jumpcorrs[tt][res][k] = mycorrelate2(powers)

    def jump2Dcorr(self, dihs=['phi:', 'psi:', 'omega:'], deltas=[0, -1, 0]):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        if not hasattr(self, 'jumpcorrs'):
            self.jumpcorrs=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, classified in self.classdih.items():
            if tt not in self.jumpcorrs.keys():
                self.jumpcorrs[tt]=dict()
            for res in res_list:
                if res not in self.jumpcorrs[tt].keys():
                    self.jumpcorrs[tt][res]=dict()
                dsum = ''
                for d in dihs:
                    dsum=dsum+d
                self.jumpcorrs[tt][res][dsum]=dict()
                labels=[el+str(res+deltas[n]) for n, el in enumerate(dihs)]
                if len(dihs)==2:
                    angles = [classified[l] for l in labels if l in classified.keys()]
                    ave_angles = [[self.ave_dih[tt][labels[n]][c] for c in values] for n, values in enumerate(angles)]
                else:
                    #this is a 2D GAF! if more than two rotations are present,
                    #we assume the first N to be co-axial
                    temp = [classified[l] for l in labels[:-1] if l in classified.keys()]
                    temp_ave = np.sum([[self.ave_dih[tt][labels[n]][c] for c in values] for n, values in enumerate(temp)], axis=0)
                    ave_angles = [temp_ave, [self.ave_dih[tt][labels[-1]][c] for c in classified[labels[-1]]] ]

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            exp1 = 1j*k*np.array(ave_angles[0])*np.pi/180.+1j*l*np.array(ave_angles[1])*np.pi/180.
                            exp2 = 1j*kp*np.array(ave_angles[0])*np.pi/180.+1j*l*np.array(ave_angles[1])*np.pi/180.
                            pow1 = np.exp(exp1)
                            pow2 = np.exp(exp2)
                            self.jumpcorrs[tt][res][dsum][str(k)+str(kp)+str(l)] = mycrosscorrelate2(pow1, pow2)

    def jump2DS2(self, dihs=['phi:', 'psi:', 'omega:'], deltas=[0, -1, 0]):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        if not hasattr(self, 'jumpS2'):
            self.jumpS2=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, classified in self.classdih.items():
            if tt not in self.jumpS2.keys():
                self.jumpS2[tt]=dict()
            for res in res_list:
                if res not in self.jumpS2[tt].keys():
                    self.jumpS2[tt][res]=dict()
                dsum = ''
                for d in dihs:
                    dsum=dsum+d
                self.jumpS2[tt][res][dsum]=dict()
                labels=[el+str(res+deltas[n]) for n, el in enumerate(dihs)]
                if len(dihs)==2:
                    angles = [classified[l] for l in labels if l in classified.keys()]
                    ave_angles = [[self.ave_dih[tt][labels[n]][c] for c in values] for n, values in enumerate(angles)]
                else:
                    #this is a 2D GAF! if more than two rotations are present,
                    #we assume the first N to be co-axial
                    temp = [classified[l] for l in labels[:-1] if l in classified.keys()]
                    temp_ave = np.sum([[self.ave_dih[tt][labels[n]][c] for c in values] for n, values in enumerate(temp)], axis=0)
                    ave_angles = [temp_ave, [self.ave_dih[tt][labels[-1]][c] for c in classified[labels[-1]]] ]

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            exp1 = 1j*k*np.array(ave_angles[0])*np.pi/180.+1j*l*np.array(ave_angles[1])*np.pi/180.
                            exp2 = 1j*kp*np.array(ave_angles[0])*np.pi/180.+1j*l*np.array(ave_angles[1])*np.pi/180.
                            pow1 = np.exp(exp1)
                            pow2 = np.exp(exp2)
                            self.jumpS2[tt][res][dsum][str(k)+str(kp)+str(l)] = np.real(np.mean(pow1)*np.conjugate(np.mean(pow2)))

    def jump1Dcorr(self, dihs='omega:', delta=0):
        """
        See Bremi Bruschweiler Ernst JACS 1997 and Bremi Bruschweiler JACS 1997.
        """
        if not hasattr(self, 'jumpcorrs'):
            self.jumpcorrs=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, classified in self.classdih.items():
            if tt not in self.jumpcorrs.keys():
                self.jumpcorrs[tt]=dict()
            for res in res_list:
                if res not in self.jumpcorrs[tt].keys():
                    self.jumpcorrs[tt][res]=dict()
                self.jumpcorrs[tt][res][dihs]=dict()
                label=dihs+str(res+delta)
                angles = classified[label]
                ave_angles = [self.ave_dih[tt][label][c] for c in angles]

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        exp1 = -1j*k*np.array(ave_angles)*np.pi/180.
                        exp2 = -1j*kp*np.array(ave_angles)*np.pi/180.
                        pow1 = np.exp(exp1)
                        pow2 = np.exp(exp2)
                        self.jumpcorrs[tt][res][dihs][str(k)+str(kp)] = mycrosscorrelate2(pow1, pow2)

    def jump1DS2(self, dihs='omega:', delta=0):
        """
        See Bremi Bruschweiler Ernst JACS 1997 and Bremi Bruschweiler JACS 1997.
        """
        if not hasattr(self, 'jumpS2'):
            self.jumpS2=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, classified in self.classdih.items():
            if tt not in self.jumpS2.keys():
                self.jumpS2[tt]=dict()
            for res in res_list:
                if res not in self.jumpS2[tt].keys():
                    self.jumpS2[tt][res]=dict()
                self.jumpS2[tt][res][dihs]=dict()
                label=dihs+str(res+delta)
                angles = classified[label]
                ave_angles = [self.ave_dih[tt][label][c] for c in angles]

                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        exp1 = -1j*k*np.array(ave_angles)*np.pi/180.
                        exp2 = -1j*kp*np.array(ave_angles)*np.pi/180.
                        pow1 = np.exp(exp1)
                        pow2 = np.exp(exp2)
                        self.jumpS2[tt][res][dihs][str(k)+str(kp)] = np.real(np.mean(pow1)*np.conjugate(np.mean(pow2)))

    def Y2ave(self, masks=['@CA', '@N', '@HN']):

        self.aveY2=dict()
        self.aveY2star=dict()

        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k]!='PRO'])
        res_list = [el for el in res_list[1:]]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top)
            self.aveY2[tt]=dict()
            self.aveY2star[tt]=dict()
            #read angles
            for n, res in enumerate(res_list):
                selection = []
                for i in range(3):
                    temp = pt.select_atoms(traj.top, ':'+str(res)+masks[i])[0]
                    selection.append(temp)
                self.aveY2[tt][res]=dict()
                self.aveY2star[tt][res]=dict()
                angles = pt.angle(traj, selection)[0]
                polars = []
                azims = []
                for angle in angles:
                    r, polar, azim = cart2sph(0., np.cos(angle*np.pi/180.), np.sin(angle*np.pi/180.))
                    polars.append(polar)
                    azims.append(azim)
                for k in range(-2, 3):
                    Yk = [Y(2, k, azim, polar) for azim, polar in zip(azims, polars)]
                    self.aveY2[tt][res][k] = np.mean(Yk)
                    self.aveY2star[tt][res][k] = np.mean(np.conjugate(Yk))

    def ave_angle(self, masks=['@C', '@N', '@CA'], deltas=[-1, 0, 0], stride=2):

        aveb=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k]!='PRO'])
        res_list = [el for el in res_list[1:]]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            aveb[tt]=dict()
            #read angles
            for n, res in enumerate(res_list):
                selection = []
                for i in range(3):
                    temp = pt.select_atoms(traj.top, ':'+str(res+deltas[i])+masks[i])[0]
                    selection.append(temp)
                angles = pt.angle(traj, selection)[0]
                aveb[tt][res] = np.mean(angles)
        return aveb

    def get_angle(self, masks=['@C', '@N', '@CA'], deltas=[-1, 0, 0]):

        ang=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k]!='PRO'])
        res_list = [el for el in res_list[1:]]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top)
            ang[tt]=dict()
            #read angles
            for n, res in enumerate(res_list):
                selection = []
                for i in range(3):
                    temp = pt.select_atoms(traj.top, ':'+str(res+deltas[i])+masks[i])[0]
                    selection.append(temp)
                ang[tt][res] = pt.angle(traj, selection)[0]
        return ang

    def ave_angle_vectors(self, masks=[['@N', '@CA'], ['@CA', '@C']], deltas=[[0, 0], [-1, -1]], stride=2):

        aveb=dict()

        res_list = sorted([k for k in self.sequence.keys()])
        res_list = [el for el in res_list[1:]]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            aveb[tt]=dict()
            #read vectors
            for n, res in enumerate(res_list):
                indices = [[pt.select_atoms(traj.top, ':'+str(res+deltas[ns][nn])+m)[0] for nn, m in enumerate(ms)] for ns, ms in enumerate(masks)]
                data_vec = [va.vector_mask(traj, [ids], dtype='ndarray') for ids in indices]
                cosines = np.array([np.dot(unit_vector(v1), unit_vector(v2)) for v1, v2 in zip(data_vec[0], data_vec[1])])
                angles = np.arccos(cosines)
                aveb[tt][res] = np.mean(angles*180./np.pi)
        return aveb

    def simplefullcorr(self, angle=118.2):
        """
        here by 'simple' we mean that we calculate only the terms in which k=k'=l, i.e.
        we assume that the two axis are collinear (good approximation for phi/psi rotations,
        beta=176.1 degrees). See Bremi Bruschweiler Ernst JACS 1997.
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        r, polar, azim = cart2sph(0., np.cos(angle*np.pi/180.), np.sin(angle*np.pi/180.))

        self.totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.jumpcorrs:
            for res in self.jumpcorrs[tt].keys():
                all_corrs = [
                    0.8*np.pi*self.gafcorrs[tt][res][k]*self.jumpcorrs[tt][res][k]*Y(2, k, azim, polar)*np.conjugate(Y(2, k, azim, polar))
                    for k in range(-2, 3)]
                temp = np.sum(all_corrs, axis=0)
                self.totcorrs[tt][res] = np.real(temp) #/max(temp)

    def full2Dcorr(self, angle=118.2, beta=121.9):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        r, polar, azim = cart2sph(0., np.cos(angle*np.pi/180.), np.sin(angle*np.pi/180.))

        beta=angle*np.pi/180.

        self.totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.jumpcorrs:
            for res in self.jumpcorrs[tt].keys():
                all_corrs = []
                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(k)+str(kp)+str(l)
                            temp = self.gafcorrs[tt][res][lab]*self.jumpcorrs[tt][res][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                            all_corrs.append(temp)
                self.totcorrs[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
                self.totcorrs[tt][res] = self.totcorrs[tt][res][:int(np.floor(len(run.totcorrs[tt][res])/2))]/self.totcorrs[tt][res][0]

    def corr2DGAF(self, omega=118.2, beta=3.9, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        r, polar, azim = cart2sph(0., np.cos(omega*np.pi/180.), np.sin(omega*np.pi/180.))

        beta=beta*np.pi/180.

        totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.jumpcorrs:
            for res in self.jumpcorrs[tt].keys():
                all_corrs = []
                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(k)+str(kp)+str(l)
                            temp = self.gafcorrs[tt][res][dihs][lab]*self.jumpcorrs[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                            all_corrs.append(temp)
                totcorrs[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
                totcorrs[tt][res] = totcorrs[tt][res]/totcorrs[tt][res][0]
        return totcorrs

    def ordpar2DGAF(self, omega=118.2, beta=3.9, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        r, polar, azim = cart2sph(0., np.cos(omega*np.pi/180.), np.sin(omega*np.pi/180.))

        beta=beta*np.pi/180.

        S2 = {tt:dict() for tt in self.jumpS2}

        for tt in self.jumpS2:
            for res in self.jumpS2[tt].keys():
                all_corrs = []
                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(k)+str(kp)+str(l)
                            temp = self.gafS2[tt][res][dihs][lab]*self.jumpS2[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                            all_corrs.append(temp)
                S2[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
        return S2

    def ordpar2DGAFv2(self, omegas, betas, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        this differs from v1 because values of angles are read from the trajectories
        omega and beta are dictionaries obtained from ave_angle and ave_angle_vectors
        omega is the angle between the vector of interest and the 1st axis of rotation
        beta is the angle between the two axes
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        S2 = {tt:dict() for tt in self.jumpS2}

        for tt in self.jumpS2:
            for res in self.jumpS2[tt].keys():
                if self.sequence[res] != 'PRO':
                    beta=betas[tt][res]*np.pi/180.
                    r, polar, azim = cart2sph(0., np.cos(omegas[tt][res]*np.pi/180.), np.sin(omegas[tt][res]*np.pi/180.))
                    all_corrs = []
                    for k in range(-2, 3):
                        for kp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(k)+str(kp)+str(l)
                                temp = self.gafS2[tt][res][dihs][lab]*self.jumpS2[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                                all_corrs.append(temp)
                    S2[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
        return S2

    def ordpar2DGAFv2_nojumps(self, omegas, betas, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        this differs from v1 because values of angles are read from the trajectories
        omega and beta are dictionaries obtained from ave_angle and ave_angle_vectors
        omega is the angle between the vector of interest and the 1st axis of rotation
        beta is the angle between the two axes
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        S2 = {tt:dict() for tt in self.gafS2}

        for tt in self.gafS2:
            for res in self.gafS2[tt].keys():
                if self.sequence[res] != 'PRO':
                    beta=betas[tt][res]*np.pi/180.
                    r, polar, azim = cart2sph(0., np.cos(omegas[tt][res]*np.pi/180.), np.sin(omegas[tt][res]*np.pi/180.))
                    all_corrs = []
                    for k in range(-2, 3):
                        for kp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(k)+str(kp)+str(l)
                                temp = self.gafS2[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                                all_corrs.append(temp)
                    S2[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
        return S2

    def corr2DGAFv2(self, omegas, betas, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        this differs from v1 because values of angles are read from the trajectories
        omega and beta are dictionaries obtained from ave_angle and ave_angle_vectors
        omega is the angle between the vector of interest and the 1st axis of rotation
        beta is the angle between the two axes
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.jumpcorrs:
            for res in self.jumpcorrs[tt].keys():
                if self.sequence[res] != 'PRO':
                    beta=betas[tt][res]*np.pi/180.
                    r, polar, azim = cart2sph(0., np.cos(omegas[tt][res]*np.pi/180.), np.sin(omegas[tt][res]*np.pi/180.))
                    all_corrs = []
                    for k in range(-2, 3):
                        for kp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(k)+str(kp)+str(l)
                                temp = self.gafcorrs[tt][res][dihs][lab]*self.jumpcorrs[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                                all_corrs.append(temp)
                    totcorrs[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
                    totcorrs[tt][res] = totcorrs[tt][res]/totcorrs[tt][res][0]
        return totcorrs

    def corr2DGAFv2_nojumps(self, omegas, betas, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        this differs from v1 because values of angles are read from the trajectories
        omega and beta are dictionaries obtained from ave_angle and ave_angle_vectors
        omega is the angle between the vector of interest and the 1st axis of rotation
        beta is the angle between the two axes
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        totcorrs = {tt:dict() for tt in self.gafcorrs}

        for tt in self.gafcorrs:
            for res in self.gafcorrs[tt].keys():
                if self.sequence[res] != 'PRO':
                    beta=betas[tt][res]*np.pi/180.
                    r, polar, azim = cart2sph(np.sin(omegas[tt][res]*np.pi/180.), 0., np.cos(omegas[tt][res]*np.pi/180.))
                    all_corrs = []
                    for k in range(-2, 3):
                        for kp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(k)+str(kp)+str(l)
                                temp = self.gafcorrs[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                                all_corrs.append(temp)
                    totcorrs[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
                    totcorrs[tt][res] = totcorrs[tt][res]/totcorrs[tt][res][0]
        return totcorrs

    def corr2DGAF_nojumps(self, omega, beta, dihs='phi:psi:'):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        omega is the angle between the vector of interest and the 1st axis of rotation
        beta is the angle between the two axes
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        totcorrs = {tt:dict() for tt in self.gafcorrs}

        for tt in self.gafcorrs:
            for res in self.gafcorrs[tt].keys():
                if self.sequence[res] != 'PRO':
                    beta=beta*np.pi/180.
                    r, polar, azim = cart2sph(np.sin(omega*np.pi/180.), 0., np.cos(omega*np.pi/180.))
                    all_corrs = []
                    for k in range(-2, 3):
                        for kp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(k)+str(kp)+str(l)
                                temp = self.gafcorrs[tt][res][dihs][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))
                                all_corrs.append(temp)
                    totcorrs[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
                    totcorrs[tt][res] = totcorrs[tt][res]/totcorrs[tt][res][0]
        return totcorrs

    def add1DGAF(self, corrs, beta=115.6, dihs='omega:'):
        """
        See Bremi Bruschweiler JACS 1997.
        """

        beta=beta*np.pi/180.

        totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.jumpcorrs:
            for res in self.jumpcorrs[tt].keys():
                if self.sequence[res] != 'PRO':
                    all_corrs = []
                    for m in range(-2, 3):
                        for mp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(m)+str(mp)
                                temp = self.gafcorrs[tt][res][dihs][lab]*self.jumpcorrs[tt][res][dihs][lab]*wignerd(2,m,l)(beta)*wignerd(2,mp,l)(beta)
                                all_corrs.append(temp)
                    temp = np.real(np.sum(all_corrs, axis=0))
                    totcorrs[tt][res] = (temp/temp[0])*corrs[tt][res]
        return totcorrs

    def add1DGAF_nojumps(self, corrs, beta=115.6, dihs='omega:'):
        """
        See Bremi Bruschweiler JACS 1997.
        """

        beta=beta*np.pi/180.

        totcorrs = {tt:dict() for tt in self.gafcorrs}

        for tt in self.gafcorrs:
            for res in self.gafcorrs[tt].keys():
                if self.sequence[res] != 'PRO':
                    all_corrs = []
                    for m in range(-2, 3):
                        for mp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(m)+str(mp)
                                temp = self.gafcorrs[tt][res][dihs][lab]*wignerd(2,m,l)(beta)*wignerd(2,mp,l)(beta)
                                all_corrs.append(temp)
                    temp = np.real(np.sum(all_corrs, axis=0))
                    totcorrs[tt][res] = (temp/temp[0])*corrs[tt][res]
        return totcorrs

    def add1DGAFS2(self, oldS2, beta=115.6, dihs='omega:'):
        """
        See Bremi Bruschweiler JACS 1997.
        """

        beta=beta*np.pi/180.

        S2 = {tt:dict() for tt in self.jumpS2}

        for tt in self.jumpS2:
            for res in self.jumpS2[tt].keys():
                if self.sequence[res] != 'PRO':
                    all_corrs = []
                    for m in range(-2, 3):
                        for mp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(m)+str(mp)
                                temp = self.gafS2[tt][res][dihs][lab]*self.jumpS2[tt][res][dihs][lab]*wignerd(2,m,l)(beta)*wignerd(2,mp,l)(beta)
                                all_corrs.append(temp)
                    temp = np.real(np.sum(all_corrs, axis=0))
                    S2[tt][res] = temp*oldS2[tt][res]*0.2
        return S2

    def add1DGAFS2_nojumps(self, oldS2, beta=115.6, dihs='omega:'):
        """
        See Bremi Bruschweiler JACS 1997.
        """

        beta=beta*np.pi/180.

        S2 = {tt:dict() for tt in self.gafS2}

        for tt in self.gafS2:
            for res in self.gafS2[tt].keys():
                if self.sequence[res] != 'PRO':
                    all_corrs = []
                    for m in range(-2, 3):
                        for mp in range(-2, 3):
                            for l in range(-2, 3):
                                lab=str(m)+str(mp)
                                temp = self.gafS2[tt][res][dihs][lab]*wignerd(2,m,l)(beta)*wignerd(2,mp,l)(beta)
                                all_corrs.append(temp)
                    temp = np.real(np.sum(all_corrs, axis=0))
                    S2[tt][res] = temp*oldS2[tt][res]*0.2
        return S2

    def add1DGAFv2(self, corrs, betas, dihs='omega:'):
        """
        this differs from v1 because values of angles are read from the trajectories
        beta is a dictionary obtained from ave_angle (or ave_angle_vectors)
        beta is the the current rotation axis and the molecular frame or the n-1 system

        See Bremi Bruschweiler JACS 1997.
        """

        totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.jumpcorrs:
            for res in self.jumpcorrs[tt].keys():
                beta=betas[tt][res]*np.pi/180.
                all_corrs = []
                for m in range(-2, 3):
                    for mp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(m)+str(mp)
                            temp = self.gafcorrs[tt][res][dihs][lab]*self.jumpcorrs[tt][res][dihs][lab]*wignerd(2,m,l)(beta)*wignerd(2,mp,l)(beta)
                            all_corrs.append(temp)
                temp = np.real(np.sum(all_corrs, axis=0))
                totcorrs[tt][res] = (temp/temp[0])*corrs[tt][res]
        return totcorrs

    def add1DGAFv2_nojumps(self, corrs, betas, dihs='omega:'):
        """
        this differs from v1 because values of angles are read from the trajectories
        beta is a dictionary obtained from ave_angle (or ave_angle_vectors)
        beta is the the current rotation axis and the molecular frame or the n-1 system

        See Bremi Bruschweiler JACS 1997.
        """

        totcorrs = {tt:dict() for tt in self.gafcorrs}

        for tt in self.gafcorrs:
            #for res in self.gafcorrs[tt].keys():
             for res in betas[tt].keys():
                beta=betas[tt][res]*np.pi/180.
                all_corrs = []
                for m in range(-2, 3):
                    for mp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(m)+str(mp)
                            temp = self.gafcorrs[tt][res][dihs][lab]*wignerd(2,m,l)(beta)*wignerd(2,mp,l)(beta)
                            all_corrs.append(temp)
                temp = np.real(np.sum(all_corrs, axis=0))
                totcorrs[tt][res] = (temp/temp[0])*corrs[tt][res]
        return totcorrs

    def corr1DGAFv2_nojumps(self, betas, dihs='psi:'):

        totcorrs = {tt:dict() for tt in self.gafcorrs}

        for tt in self.gafcorrs:
            for res in self.gafcorrs[tt].keys():
                beta=betas[tt][res]*np.pi/180.
                all_corrs = []
                for m in range(-2, 3):
                    for mp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(m)+str(mp)
                            temp = self.gafcorrs[tt][res][dihs][lab]*wignerd(2,m,l)(np.pi-beta)*wignerd(2,mp,l)(np.pi-beta)
                            all_corrs.append(temp)
                temp = np.real(np.sum(all_corrs, axis=0))
                totcorrs[tt][res] = temp/temp[0]
        return totcorrs

    def corr1DGAF_nojumps(self, beta, dihs='psi:'):

        totcorrs = {tt:dict() for tt in self.gafcorrs}

        for tt in self.gafcorrs:
            for res in self.gafcorrs[tt].keys():
                beta=beta*np.pi/180.
                all_corrs = []
                for m in range(-2, 3):
                    for mp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(m)+str(mp)
                            temp = self.gafcorrs[tt][res][dihs][lab]*wignerd(2,m,l)(np.pi-beta)*wignerd(2,mp,l)(np.pi-beta)
                            all_corrs.append(temp)
                temp = np.real(np.sum(all_corrs, axis=0))
                totcorrs[tt][res] = temp/temp[0]
        return totcorrs

    def full2DS2(self, angle=118.2, beta=121.9):
        """
        See Bremi Bruschweiler Ernst JACS 1997.
        """
        Y = lambda l, m, azim, polar: sph_harm(m, l, azim, polar)

        r, polar, azim = cart2sph(0., np.cos(angle*np.pi/180.), np.sin(angle*np.pi/180.))

        beta=angle*np.pi/180.

        self.S2 = {tt:dict() for tt in self.aveb}

        for tt in self.aveb:
            for res in self.aveb[tt].keys():
                all_s2 = []
                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            Fkl = np.exp(-0.5*((l**2)*self.vardih[tt][res][0]+(k**2)*self.vardih[tt][res][1]+2*k*l*np.sqrt(self.vardih[tt][res][0]*self.vardih[tt][res][1])*self.rdih[tt][res]))
                            Fkpl = np.exp(-0.5*((l**2)*self.vardih[tt][res][0]+(kp**2)*self.vardih[tt][res][1]+2*kp*l*np.sqrt(self.vardih[tt][res][0]*self.vardih[tt][res][1])*self.rdih[tt][res]))
                            temp = wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*Y(2, k, azim, polar)*np.conjugate(Y(2, kp, azim, polar))*Fkl*Fkpl
                            all_s2.append(temp)
                self.S2[tt][res] = 0.8*np.pi*np.real(np.sum(all_s2))

    def full2DS2_v2(self):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        the difference between this and v1 is that here we use the ensemble averages of Y2m and beta
        """

        self.S2 = {tt:dict() for tt in self.aveb}

        for tt in self.aveb:
            for res in self.aveb[tt].keys():
                all_s2 = []
                beta = self.aveb[tt][res]*np.pi/180.
                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            Fkl = np.exp(-0.5*((l**2)*self.vardih[tt][res][0]+(k**2)*self.vardih[tt][res][1])+2*k*l*np.sqrt(self.vardih[tt][res][0]*self.vardih[tt][res][1])*self.rdih[tt][res])
                            Fkpl = np.exp(-0.5*((l**2)*self.vardih[tt][res][0]+(kp**2)*self.vardih[tt][res][1])+2*kp*l*np.sqrt(self.vardih[tt][res][0]*self.vardih[tt][res][1])*self.rdih[tt][res])
                            temp = wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*self.aveY2[tt][res][k]*self.aveY2star[tt][res][kp]*Fkl*Fkpl
                            all_s2.append(temp)
                self.S2[tt][res] = 0.8*np.pi*np.real(np.sum(all_s2))

    def full2Dcorr2(self):
        """
        See Bremi Bruschweiler Ernst JACS 1997.

        the difference between this and v1 is that here we use the ensemble averages of Y2m and beta
        """

        self.totcorrs = {tt:dict() for tt in self.jumpcorrs}

        for tt in self.aveY2.keys():
            for res in self.aveY2[tt].keys():
                beta = self.aveb[tt][res]*np.pi/180.
                all_corrs = []
                for k in range(-2, 3):
                    for kp in range(-2, 3):
                        for l in range(-2, 3):
                            lab=str(k)+str(kp)+str(l)
                            temp = self.gafcorrs[tt][res][lab]*self.jumpcorrs[tt][res][lab]*wignerd(2,k,l)(np.pi-beta)*wignerd(2,kp,l)(np.pi-beta)*self.aveY2[tt][res][k]*self.aveY2star[tt][res][kp]
                            all_corrs.append(temp)
                self.totcorrs[tt][res] = 0.8*np.pi*np.real(np.sum(all_corrs, axis=0))
                self.totcorrs[tt][res] = self.totcorrs[tt][res][:int(np.floor(len(self.totcorrs[tt][res])/2))]/self.totcorrs[tt][res][0]

    def simplefullcorr2(self):
        """
        here by 'simple' we mean that we calculate only the terms in which k=k'=l, i.e.
        we assume that the two axis are collinear (good approximation for phi/psi rotations,
        beta=176.1 degrees). See Bremi Bruschweiler Ernst JACS 1997.

        the difference between this and v1 is that here we use the ensemble averages of Y2m
        """

        self.totcorrs = {tt:dict() for tt in self.aveY2.keys()}

        for tt in self.aveY2.keys():
            for res in self.aveY2[tt].keys():
                all_corrs = [
                    0.8*np.pi*self.gafcorrs[tt][res][k]*self.jumpcorrs[tt][res][k]*self.aveY2[tt][res][k]*self.aveY2star[tt][res][k]
                    for k in range(-2, 3)]
                temp = np.sum(all_corrs, axis=0)
                self.totcorrs[tt][res] = np.real(temp) #/max(temp)

    def fullNHflucts(self, stride=2):
        self.NHflucts=dict()
        self.NHcorrs=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k] != 'PRO'])
        res_list = res_list[1:]

        for tt, t in enumerate(self.trajs):
            traj = pt.load(t, self.top, stride=stride)
            self.NHflucts[tt]=dict()
            self.NHcorrs[tt]=dict()
            for r in res_list:
                new_traj = pt.superpose(traj, ref=0, mask='(@CA,C,O) & (:'+str(r-1)+')')
                #read vectors
                self.n_indices = pt.select_atoms('@N & (:'+str(r)+')', new_traj.top)
                self.h_indices = self.n_indices + 1
                self.nh_pairs = np.array(list(zip(self.n_indices, self.h_indices)))
                vals = va.vector_mask(new_traj, self.nh_pairs, dtype='ndarray')
                self.NHcorrs[tt][r] = pt.timecorr(vals, vals, order=2, tstep=1, tcorr=len(vals)/2., norm=False, dtype='ndarray')
                self.NHflucts[tt][r] = S2(vals)

    def fullCANflucts(self, stride=2):
        self.CANflucts=dict()
        self.CANcorrs=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k] != 'PRO'])
        res_list = res_list[1:]

        for tt, t in enumerate(self.trajs):
            traj = pt.load(t, self.top, stride=stride)
            self.CANflucts[tt]=dict()
            self.CANcorrs[tt]=dict()
            for r in res_list:
                new_traj = pt.superpose(traj, ref=0, mask='(@CA,C,O) & (:'+str(r-1)+')')
                #read vectors
                self.n_indices = pt.select_atoms('@N & (:'+str(r)+')', new_traj.top)
                self.ca_indices = pt.select_atoms('@CA & (:'+str(r)+')', new_traj.top)
                self.can_pairs = np.array(list(zip(self.ca_indices, self.n_indices)))
                vals = va.vector_mask(new_traj, self.can_pairs, dtype='ndarray')
                self.CANcorrs[tt][r] = pt.timecorr(vals, vals, order=2, tstep=1, tcorr=len(vals)/2., norm=False, dtype='ndarray')
                self.CANflucts[tt][r] = S2(vals)

    def ordparNHoop(self, stride=2):
        """
        oop = out-of-plane
        """
        self.NHoopS2=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k] != 'PRO'])
        res_list = res_list[1:]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            reshere = []
            for r in self.sequence.keys():
                if r+1 in self.sequence.keys():
                    if self.sequence[r+1] != 'PRO':
                        reshere.append(r)
            indices1 = []
            indices2 = []
            for r in reshere:
                indices1.append(pt.select_atoms('@CA  & :'+str(r), traj.top)[0])
                indices2.append(pt.select_atoms('@C  & :'+str(r), traj.top)[0])
            #indices1 = pt.select_atoms('@CA', traj.top)[:-1]
            #indices2 = pt.select_atoms('@C', traj.top)[:-1]
            #indices3 = pt.select_atoms('@N'+' & !(:1)', traj.top)
            indices3 = pt.select_atoms('@N  & !(:1) & !(:PRO)', traj.top)
            pairs12 = np.array(list(zip(indices1, indices2)))
            pairs23 = np.array(list(zip(indices2, indices3)))
            data_vec12 = va.vector_mask(traj, pairs12, dtype='ndarray')
            data_vec23 = va.vector_mask(traj, pairs23, dtype='ndarray')
            vectors = np.array([np.cross(vals12, vals23) for vals12, vals23 in zip(data_vec12, data_vec23)])
            #these are the vectors perpendicular to the peptide planes
            n_indices = pt.select_atoms('@N & !(:1) & !(:PRO)', traj.top)
            h_indices = n_indices + 1
            nh_pairs = np.array(list(zip(n_indices, h_indices)))
            nh_vecs = va.vector_mask(traj, nh_pairs, dtype='ndarray')
            self.NHoopS2[tt]=dict()
            for n, res in enumerate(res_list):
                vec_p = vectors[n]
                vec_nh = nh_vecs[n]
                cosines = np.array([np.dot(unit_vector(v1), unit_vector(v2)) for v1, v2 in zip(vec_p, vec_nh)])
                angles = np.arccos(cosines)+np.pi/2
                cosines = np.cos(angles)
                self.NHoopS2[tt][res] = 1.5*np.mean(cosines)**2-0.5

    def ordparNHip(self, stride=2):
        """
        oop =in-plane
        this corresponds to fluctuations of the CA-N-H bond
        """
        self.NHipS2=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k] != 'PRO'])
        res_list = res_list[1:]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            self.NHipS2[tt]=dict()
            #read angles
            for n, res in enumerate(res_list):
                selection = []
                for m in ['@CA', '@N', '@HN']:
                    temp = pt.select_atoms(':'+str(res)+' & '+m, traj.top)[0]
                    selection.append(temp)
                angles = pt.angle(traj, selection)[0]
                ave = np.mean(angles)
                cosines = np.cos((angles-ave)*np.pi/180.)
                self.NHipS2[tt][res] = 1.5*np.mean(cosines)**2-0.5

    def NHoopcorrs(self):
        """
        oop = out-of-plane
        """
        self.corrsNHoop=dict()

        res_list = sorted([k for k in self.sequence.keys() if self.sequence[k] != 'PRO'])
        res_list = res_list[1:]

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top)
            reshere = []
            for r in self.sequence.keys():
                if r+1 in self.sequence.keys():
                    if self.sequence[r+1] != 'PRO':
                        reshere.append(r)
            indices1 = []
            indices2 = []
            for r in reshere:
                indices1.append(pt.select_atoms('@CA  & :'+str(r), traj.top)[0])
                indices2.append(pt.select_atoms('@C  & :'+str(r), traj.top)[0])
            #indices1 = pt.select_atoms('@CA', traj.top)[:-1]
            #indices2 = pt.select_atoms('@C', traj.top)[:-1]
            #indices3 = pt.select_atoms('@N'+' & !(:1)', traj.top)
            indices3 = pt.select_atoms('@N  & !(:1) & !(:PRO)', traj.top)
            pairs12 = np.array(list(zip(indices1, indices2)))
            pairs23 = np.array(list(zip(indices2, indices3)))
            data_vec12 = va.vector_mask(traj, pairs12, dtype='ndarray')
            data_vec23 = va.vector_mask(traj, pairs23, dtype='ndarray')
            vectors = np.array([np.cross(vals12, vals23) for vals12, vals23 in zip(data_vec12, data_vec23)])
            #these are the vectors perpendicular to the peptide planes
            n_indices = pt.select_atoms('@N & !(:1) & !(:PRO)', traj.top)
            h_indices = n_indices + 1
            nh_pairs = np.array(list(zip(n_indices, h_indices)))
            nh_vecs = va.vector_mask(traj, nh_pairs, dtype='ndarray')
            self.corrsNHoop[tt]=dict()
            for n, res in enumerate(res_list):
                vec_p = vectors[n]
                vec_nh = nh_vecs[n]
                cosines = np.array([np.dot(unit_vector(v1), unit_vector(v2)) for v1, v2 in zip(vec_p, vec_nh)])
                angles = np.arccos(cosines)+np.pi/2
                cosines = np.cos(angles)
                self.corrsNHoop[tt][res] = mycorrelate2(1.5*cosines**2-0.5)


class labdyn:

    def __init__(self, top, trajs):
        #read sequence
        traj = pt.iterload(trajs[0], top)
        self.sequence = {res.index+1:res.name for res in traj.top.residues}
        self.top = top
        self.trajs = trajs

    def caca(self, rotfit=False, stride=2):
        self.acf=dict()
        #self.ccf=dict()

        res_list = sorted([k for k, val in self.sequence.items() if k != 1])

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            if rotfit == True:
                traj = pt.load(t, self.top, stride=stride)
                #print('RMSD fitting on '+mask)
                _ = traj.superpose(ref=-1, mask='@CA')
            #read vectors
            self.ca_indices = pt.select_atoms('@CA', traj.top)
            self.ca_pairs = np.array(list(zip(self.ca_indices, self.ca_indices[1:])))
            data_vec = va.vector_mask(traj, self.ca_pairs, dtype='ndarray')
            self.acf[tt] = {res_list[n]:pt.timecorr(vals, vals, order=2, tstep=1, tcorr=len(vals)/2., norm=False, dtype='ndarray')
                for n, vals in enumerate(data_vec)}
            # self.ccf[tt] = {res_list[n]:{res_list[m]:pt.timecorr(vals, vals2, order=2, tstep=1, tcorr=len(vals)/2., norm=False, dtype='ndarray') for m, vals2 in enumerate(data_vec) if m!=n}
            #     for n, vals in enumerate(data_vec)}

    def caca_vecs(self, rotfit=False, stride=2):

        res_list = sorted([k for k, val in self.sequence.items() if k != 1])

        vecs = dict()

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            if rotfit == True:
                traj = pt.load(t, self.top, stride=stride)
                #print('RMSD fitting on '+mask)
                _ = traj.superpose(ref=-1, mask='@CA')
            #read vectors
            self.ca_indices = pt.select_atoms('@CA', traj.top)
            self.ca_pairs = np.array(list(zip(self.ca_indices, self.ca_indices[1:])))
            vecs[tt] = va.vector_mask(traj, self.ca_pairs, dtype='ndarray')
            self.vecs = vecs
        return vecs

    def caca_corrs(self):
        res_list = sorted([k for k, val in self.sequence.items() if k != 1])
        corrs = dict()
        for tt, t in enumerate(self.trajs):
            corrs[tt] =dict()
            u_vec = [vecs/np.sqrt((vecs ** 2).sum(-1))[..., np.newaxis] for vecs in self.vecs[tt]]
            for n, r1 in enumerate(res_list):
                 corrs[tt][r1] = dict()
                 v1 = u_vec[n]
                 for m, r2 in enumerate(res_list):
                     v2 = u_vec[m]
                     

    def caca_anisotropy(self, rotfit=False, stride=2):

        res_list = sorted([k for k, val in self.sequence.items() if k != 1])

        axes = ['x', 'y', 'z']

        corrs = dict()

        for tt, t in enumerate(self.trajs):
            corrs[tt] = dict()
            traj = pt.iterload(t, self.top, stride=stride)
            if rotfit == True:
                traj = pt.load(t, self.top, stride=stride)
                #print('RMSD fitting on '+mask)
                _ = traj.superpose(ref=-1, mask='@CA')
            #read vectors
            self.ca_indices = pt.select_atoms('@CA', traj.top)
            self.ca_pairs = np.array(list(zip(self.ca_indices, self.ca_indices[1:])))
            vecs = va.vector_mask(traj, self.ca_pairs, dtype='ndarray')
            for n, values in enumerate(vecs):
                corrs[tt][res_list[n]] = dict()
                vals = np.array([unit_vector(k) for k in values])
                for i in [0, 1, 2]:
                    vali = vals[:, i]
                    for j in range(i, 3):
                        valj = vals[:, j]
                        x = vali*valj
                        corrs[tt][res_list[n]][axes[i]+axes[j]] = mycorrelate2(x, norm=False)
        return corrs

    def axes_tumbling(self, rotfit=False, stride=2):

        res_list = sorted([k for k, val in self.sequence.items() if k != 1])

        axes = ['x', 'y', 'z']

        corrs = dict()

        for tt, t in enumerate(self.trajs):
            corrs[tt] = dict()
            traj = pt.load(t, self.top, stride=stride)
            if rotfit == True:
                #print('RMSD fitting on '+mask)
                _ = traj.superpose(ref=-1, mask='@CA')
            ca_pos = traj['@CA'].xyz[:, 1:, :]
            n_pos = traj['@N'].xyz[:, 1:, :]
            c_pos = traj['@C'].xyz[:, :, :]
            #read CAC vectors
            can_vec = ca_pos-n_pos
            can_vec = np.swapaxes(can_vec,0,1)
            #z-axes of the molecular frame
            z_vec = can_vec
            #normalize
            for n in range(len(z_vec)):
                for m in range(len(z_vec[0])):
                    z_vec[n][m] = unit_vector(z_vec[n][m])
            #calculate correlation function
            for n, val in enumerate(res_list):
                corrs[tt][val] = dict()
                corrs[tt][val]['z'] = pt.timecorr(z_vec[n], z_vec[n], order=2, tstep=1, tcorr=len(z_vec[n])/2., norm=False, dtype='ndarray')
            con_vec = c_pos[:, :-1, :]-n_pos
            con_vec = np.swapaxes(con_vec,0,1)
            #y-axes of the molecular frame
            y_vec = np.cross(can_vec, con_vec)
            #normalize
            for n in range(len(y_vec)):
                for m in range(len(y_vec[0])):
                    y_vec[n][m] = unit_vector(y_vec[n][m])
            #calculate correlation function
            for n, val in enumerate(res_list):
                corrs[tt][val]['y'] = pt.timecorr(y_vec[n], y_vec[n], order=2, tstep=1, tcorr=len(y_vec[n])/2., norm=False, dtype='ndarray')
            #x-axes of the molecular frame
            x_vec = np.cross(z_vec, y_vec)
            #normalize
            for n in range(len(x_vec)):
                for m in range(len(x_vec[0])):
                    x_vec[n][m] = unit_vector(x_vec[n][m])
            #calculate correlation function
            for n, val in enumerate(res_list):
                corrs[tt][val]['x'] = pt.timecorr(x_vec[n], x_vec[n], order=2, tstep=1, tcorr=len(x_vec[n])/2., norm=False, dtype='ndarray')

        return corrs

    def plane_tumbling(self, rotfit=False, stride=2):

        res_list = sorted([k for k, val in self.sequence.items() if k != 1])

        axes = ['x', 'y', 'z']

        corrs = dict()

        for tt, t in enumerate(self.trajs):
            corrs[tt] = dict()
            traj = pt.load(t, self.top, stride=stride)
            if rotfit == True:
                #print('RMSD fitting on '+mask)
                _ = traj.superpose(ref=-1, mask='@CA')
            ca_pos = traj['@CA'].xyz[:, 1:, :]
            n_pos = traj['@N'].xyz[:, 1:, :]
            c_pos = traj['@C'].xyz[:, :, :]
            #read CAC vectors
            can_vec = ca_pos-n_pos
            can_vec = np.swapaxes(can_vec,0,1)
            #z-axes of the molecular frame
            z_vec = can_vec
            #normalize
            for n in range(len(z_vec)):
                for m in range(len(z_vec[0])):
                    z_vec[n][m] = unit_vector(z_vec[n][m])
            con_vec = c_pos[:, :-1, :]-n_pos
            con_vec = np.swapaxes(con_vec,0,1)
            #y-axes of the molecular frame
            y_vec = np.cross(can_vec, con_vec)
            #normalize
            for n in range(len(y_vec)):
                for m in range(len(y_vec[0])):
                    y_vec[n][m] = unit_vector(y_vec[n][m])
            #calculate correlation function
            for n, val in enumerate(res_list):
                corrs[tt][val] = pt.timecorr(y_vec[n], y_vec[n], order=2, tstep=1, tcorr=len(y_vec[n])/2., norm=False, dtype='ndarray')

        return corrs

    def axes_trajs(self, rotfit=False, stride=2):

            res_list = sorted([k for k, val in self.sequence.items() if k != 1])

            axes = ['x', 'y', 'z']

            trajs = dict()

            for tt, t in enumerate(self.trajs):
                trajs[tt] = dict()
                traj = pt.load(t, self.top, stride=stride)
                if rotfit == True:
                    #print('RMSD fitting on '+mask)
                    _ = traj.superpose(ref=-1, mask='@CA')
                ca_pos = traj['@CA'].xyz[:, 1:, :]
                n_pos = traj['@N'].xyz[:, 1:, :]
                c_pos = traj['@C'].xyz[:, :, :]
                #read CAC vectors
                can_vec = ca_pos-n_pos
                can_vec = np.swapaxes(can_vec,0,1)
                #z-axes of the molecular frame
                z_vec = can_vec
                #normalize
                for n in range(len(z_vec)):
                    for m in range(len(z_vec[0])):
                        z_vec[n][m] = unit_vector(z_vec[n][m])
                #calculate correlation function
                for n, val in enumerate(res_list):
                    trajs[tt][val] = dict()
                    trajs[tt][val]['z'] = [np.dot(v, z_vec[n][0]) for v in z_vec[n]]
                con_vec = c_pos[:, :-1, :]-n_pos
                con_vec = np.swapaxes(con_vec,0,1)
                #y-axes of the molecular frame
                y_vec = np.cross(can_vec, con_vec)
                #normalize
                for n in range(len(y_vec)):
                    for m in range(len(y_vec[0])):
                        y_vec[n][m] = unit_vector(y_vec[n][m])
                #calculate correlation function
                for n, val in enumerate(res_list):
                    trajs[tt][val]['y'] = [np.dot(v, y_vec[n][0]) for v in y_vec[n]]
                #x-axes of the molecular frame
                x_vec = np.cross(z_vec, y_vec)
                #normalize
                for n in range(len(x_vec)):
                    for m in range(len(x_vec[0])):
                        x_vec[n][m] = unit_vector(x_vec[n][m])
                #calculate correlation function
                for n, val in enumerate(res_list):
                    trajs[tt][val]['x'] = [np.dot(v, x_vec[n][0]) for v in x_vec[n]]

                return trajs
