import pytraj as pt
import numpy as np
from pytraj import vector as va
from scipy.constants import mu_0, h, value, pi

gammas = {
    '@N': -2.7126 * 1e7,
    '@HN': value("proton gyromag. ratio"),
    '@CA': 6.7262*1e7,
    '@C': 6.7262*1e7,
    '@HA': value("proton gyromag. ratio"),
    '@H': value("proton gyromag. ratio")
}

bond = {
    '@N@HN': 1.02 * 1e-10,
    '@N@H': 1.02 * 1e-10,
    '@CA@HA': 1.12 * 1e-10,
    '@CA@C': 1.52 * 1e-10
}

import os
from pickle import dump

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def projection(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return v1_u - np.dot(v1_u, v2_u)*v2_u


def RDC(trajfiles, topfile, rdc_types, masks, step=1, seg=None):
    results = dict()
    for n, tr in enumerate(trajfiles):
        traj = pt.iterload(tr, topfile, frame_slice=(0, -1, step))
        if seg==None:
            residues = [r.index+1 for r in traj.topology.residues if r.name!='PRO' and r.index>0]
        else:
            residues = [r.index+1 for r in traj.topology.residues
                        if r.name!='PRO' and r.index>=1 and r.index+1>=seg[0] and r.index+1<=seg[1]]
        temp = dict()
        Aa = []
        Ar = []
        eigx = []
        eigy = []
        eigz = []
        alpha = []
        gamma = []
        beta = []

        for f, frame in enumerate(traj):
            pt.write_traj("temp.pdb", frame, top=traj.top, overwrite=True)
            if seg==None:
                os.system("pales -pdb temp.pdb > pales.out")
            else:
                os.system("pales -pdb temp.pdb -s1 "+str(seg[0])+" -sN "+str(seg[1])+" > pales.out")
            os.system("grep \"DATA EIGENVALUES (Sxx_d,Syy_d,Szz_d)\" pales.out > eigenvalues.out")
            with open('eigenvalues.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            Sxx_d=float(data[-3])
            Syy_d=float(data[-2])
            Szz_d=float(data[-1])

            Aa.append(Szz_d)
            Ar.append((2./3.)*(Sxx_d-Syy_d))

            os.system("grep \"DATA EULER_ANGLES\" pales.out > euler.out")
            with open('euler.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            alpha.append(float(data[-3]))
            beta.append(float(data[-2]))
            gamma.append(float(data[-1]))

            os.system("grep \"DATA EIGENVECTORS X_AXIS\" pales.out > eigenvalues1.out")
            with open('eigenvalues1.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            eigx.append([float(data[-3]), float(data[-2]), float(data[-1])])

            os.system("grep \"DATA EIGENVECTORS Y_AXIS\" pales.out > eigenvalues2.out")
            with open('eigenvalues2.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            eigy.append([float(data[-3]), float(data[-2]), float(data[-1])])

            os.system("grep \"DATA EIGENVECTORS Z_AXIS\" pales.out > eigenvalues3.out")
            with open('eigenvalues3.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            eigz.append([float(data[-3]), float(data[-2]), float(data[-1])])

        R = np.array(Ar)/np.array(Aa)

        #get vectors from traj
        for m, el in enumerate(rdc_types):
            if seg==None:
                indices1 = pt.select_atoms(masks[m][0]+' & !(:1) & !(:PRO)', traj.top)
                indices2 = pt.select_atoms(masks[m][1]+' & !(:1) & !(:PRO)', traj.top)
            else:

                indices1 = np.array([
                    pt.select_atoms(masks[m][0]+' & :'+str(res)+' & !(:PRO)', traj.top)[0]
                    for res in residues #range(seg[0], seg[1]+1)
                    if len(pt.select_atoms(masks[m][0]+' & :'+str(res)+' & !(:PRO)', traj.top))>0])
                indices2 = np.array([
                    pt.select_atoms(masks[m][1]+' & :'+str(res)+' & !(:PRO)', traj.top)[0]
                    for res in residues #range(seg[0], seg[1]+1)
                    if len(pt.select_atoms(masks[m][1]+' & :'+str(res)+' & !(:PRO)', traj.top))>0])
            if el=='CAHA':
                indices2 = indices1+1
            pairs = np.array(list(zip(indices1, indices2)))
            data_vec = va.vector_mask(traj, pairs, dtype='ndarray')
            #calculate theta and phi angles
            thetas = [[angle_between(vec, zaxis) for vec, zaxis in zip(veclist, eigz)] for veclist in data_vec]
            projs = [[projection(vec, zaxis) for vec, zaxis in zip(veclist, eigz)] for veclist in data_vec]
            phis = [[angle_between(vec, xaxis) for vec, xaxis in zip(veclist, eigx)] for veclist in projs]

            #calculate RDC values
            #rPQcub = [np.nanmean([(np.linalg.norm(v)*1e-10)**3 for v in veclist]) for veclist in data_vec]
            #Dmax = [-(gammas[masks[m][0]]*gammas[masks[m][1]]*h*mu_0)/(8*pi**3*rPQ) for rPQ in rPQcub]
            Dmax = -(gammas[masks[m][0]]*gammas[masks[m][1]]*h*mu_0)/(8*pi**3*bond[masks[m][0]+masks[m][1]]**3)
            # temp[el] = {residues[idx]:[Dmax[idx]*(0.5*(3*np.cos(th)**2-1)*Aa[f]+0.75*Ar[f]*np.sin(th)**2*np.cos(2*phis[idx][f]) )
            #             for f, th in enumerate(ths)]
            #                 for idx, ths in enumerate(thetas)}
            temp[el] = {residues[idx]:np.mean([Dmax*(0.5*(3*np.cos(th)**2-1)*Aa[f]+0.75*Ar[f]*np.sin(th)**2*np.cos(2*phis[idx][f]) )
                        for f, th in enumerate(ths)])
                            for idx, ths in enumerate(thetas)}
        results[n] = temp
        with open('rdc_'+str(n)+'.pkl', 'wb') as tf:
            dump(temp, tf)
    return results

def calc_alignment(trajfiles, topfile, rdc_types, masks, step=1, seg=None):
    Aa = dict()
    Ar = dict()
    eigx = dict()
    eigz = dict()
    for n, tr in enumerate(trajfiles):
        traj = pt.iterload(tr, topfile, frame_slice=(0, -1, step))
        if seg==None:
            residues = [r.index+1 for r in traj.topology.residues if r.name!='PRO' and r.index>0]
        else:
            residues = [r.index+1 for r in traj.topology.residues
                        if r.name!='PRO' and r.index>=1 and r.index+1>=seg[0] and r.index+1<=seg[1]]
        Aa_temp = []
        Ar_temp = []
        eigx_temp = []
        eigz_temp = []

        for f, frame in enumerate(traj):
            pt.write_traj("temp.pdb", frame, top=traj.top, overwrite=True)
            if seg==None:
                os.system("pales -pdb temp.pdb > pales.out")
            else:
                os.system("pales -pdb temp.pdb -s1 "+str(seg[0])+" -sN "+str(seg[1])+" > pales.out")
            os.system("grep \"DATA EIGENVALUES (Sxx_d,Syy_d,Szz_d)\" pales.out > eigenvalues.out")
            with open('eigenvalues.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            Sxx_d=float(data[-3])
            Syy_d=float(data[-2])
            Szz_d=float(data[-1])

            Aa_temp.append(Szz_d)
            Ar_temp.append((2./3.)*(Sxx_d-Syy_d))

            os.system("grep \"DATA EIGENVECTORS X_AXIS\" pales.out > eigenvalues1.out")
            with open('eigenvalues1.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            eigx_temp.append([float(data[-3]), float(data[-2]), float(data[-1])])

            os.system("grep \"DATA EIGENVECTORS Z_AXIS\" pales.out > eigenvalues3.out")
            with open('eigenvalues3.out', 'r') as tf:
                data = tf.readlines()
            data = data[0].split()
            eigz_temp.append([float(data[-3]), float(data[-2]), float(data[-1])])

        eigz[n] = eigz_temp
        eigx[n] = eigx_temp
        Aa[n] = Aa_temp
        Ar[n] = Ar_temp

    return Aa, Ar, eigx, eigz


def RDC_from_S(eigx, eigz, Aa, Ar, trajfiles, topfile, rdc_types, masks, step=1, seg=None):
    results = dict()
    for n, tr in enumerate(trajfiles):
        temp =dict()
        traj = pt.iterload(tr, topfile, frame_slice=(0, -1, step))
        if seg==None:
            residues = [r.index+1 for r in traj.topology.residues if r.name!='PRO' and r.index>0]
        else:
            residues = [r.index+1 for r in traj.topology.residues
                        if r.name!='PRO' and r.index>=1 and r.index+1>=seg[0] and r.index+1<=seg[1]]
        #get vectors from traj
        for m, el in enumerate(rdc_types):
            if seg==None:
                indices1 = pt.select_atoms(masks[m][0]+' & !(:1) & !(:PRO)', traj.top)
                indices2 = pt.select_atoms(masks[m][1]+' & !(:1) & !(:PRO)', traj.top)
            else:

                indices1 = np.array([
                    pt.select_atoms(masks[m][0]+' & :'+str(res)+' & !(:PRO)', traj.top)[0]
                    for res in residues #range(seg[0], seg[1]+1)
                    if len(pt.select_atoms(masks[m][0]+' & :'+str(res)+' & !(:PRO)', traj.top))>0])
                indices2 = np.array([
                        pt.select_atoms(masks[m][1]+' & :'+str(res)+' & !(:PRO)', traj.top)[0]
                        for res in residues #range(seg[0], seg[1]+1)
                        if len(pt.select_atoms(masks[m][1]+' & :'+str(res)+' & !(:PRO)', traj.top))>0])
            if el=='CAHA':
                indices2 = indices1+1

            pairs = np.array(list(zip(indices1, indices2)))
            data_vec = va.vector_mask(traj, pairs, dtype='ndarray')
            #calculate theta and phi angles
            thetas = [[angle_between(vec, eigz) for vec in veclist] for veclist in data_vec]
            projs = [[projection(vec, eigz) for vec in veclist] for veclist in data_vec]
            phis = [[angle_between(vec, eigx) for vec in veclist] for veclist in projs]

            Dmax = -(gammas[masks[m][0]]*gammas[masks[m][1]]*h*mu_0)/(8*pi**3*bond[masks[m][0]+masks[m][1]]**3)

            temp[el] = {residues[idx]:np.mean([Dmax*(0.5*(3*np.cos(th)**2-1)*Aa+0.75*Ar*np.sin(th)**2*np.cos(2*phis[idx][f]) )
                            for f, th in enumerate(ths)])
                                for idx, ths in enumerate(thetas)}
        results[n] = temp
        with open('rdc_'+str(n)+'.pkl', 'wb') as tf:
            dump(temp, tf)
    return results

class LA_RDC:

    def __init__(self, trajfiles, topfile, rdc_types, masks, step=1, segsize=25):
        self.trajfiles = trajfiles
        self.topfile = topfile
        self.rdc_types = rdc_types
        self.masks = masks
        self.segsize = segsize
        self.step = step

    def calc_RDC(self):
        print('Calculating RDCs...')
        values = {n:{val:dict() for val in self.rdc_types} for n, el in enumerate(self.trajfiles)}

        #get number of residues
        traj = pt.iterload(self.trajfiles[0], self.topfile)
        residues = [r.index+1 for r in traj.topology.residues if r.name!='PRO' and r.index>0]
        nresidues = len([r.index for r in traj.topology.residues])
        self.residues = residues
        self.nresidues = nresidues

        #first segment
        temp = RDC(self.trajfiles, self.topfile, self.rdc_types, self.masks,
                   step=self.step, seg=[1, self.segsize])
        #of this first segment, we take the first half:
        for n, el in enumerate(self.trajfiles):
            for val in self.rdc_types:
                for res in range(int(self.segsize/2)+1):
                    if res in values[n][val].keys():
                        values[n][val][res] = temp[n][val][res]
        #last segment
        temp = RDC(self.trajfiles, self.topfile, self.rdc_types, self.masks,
                   step=self.step, seg=[self.nresidues-self.segsize, self.nresidues])
        #of this last segment, we take the second half:
        for n, el in enumerate(self.trajfiles):
            for val in self.rdc_types:
                for res in range(int(self.nresidues-self.segsize/2)-1, self.nresidues):
                    if res in values[n][val].keys():
                        values[n][val][res] = temp[n][val][res]
        #for the missing residues, we calculate their own segment:
        for res in residues:
            if res not in values[0][self.rdc_types[0]].keys():
                temp = RDC(self.trajfiles, self.topfile, self.rdc_types, self.masks,
                           step=self.step, seg=[int(res-self.segsize/2), int(res+self.segsize/2)])
                for n, el in enumerate(self.trajfiles):
                    for val in self.rdc_types:
                        values[n][val][res] = temp[n][val][res]

        self.rdc = values

    def calc_baseline(self):
        print('Calculating baseline...')
        L = self.nresidues
        m0 = int((L+1)/2)
        a = 0.33-0.22*(1-np.exp(-0.015*L))
        b = 1.16*1e5/(L**4)
        c = 9.8-6.14*(1-np.exp(-0.021*L))
        self.B = {r:2*b*np.cosh(-a*(i+1-m0))-c for i, r in enumerate(self.residues)}

    def apply_baseline(self):
        print('Applying baseline to LAW values...')
        values = {n:{val:dict() for val in self.rdc_types}
                    for n, el in enumerate(self.trajfiles)}
        for n, el in enumerate(self.trajfiles):
            for val in self.rdc_types:
                for res in self.residues:
                    values[n][val][res] = np.mean(self.rdc[n][val][res])*np.abs(self.B[res])
        self.corrected_rdc = values
