import numpy as np
import pytraj as pt

class ramapops:

    def __init__(self, top, trajs):
        #read sequence
        traj = pt.iterload(trajs[0], top)
        self.sequence = {res.index+1:res.name for res in traj.top.residues}
        self.top = top
        self.trajs = trajs


    def getdihedrals(self, s='phi psi', stride=2):
        print('reading dihedral angles from trajectories....')
        self.dih = dict()

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            #read dihedral angles
            self.dih[tt] = pt.multidihedral(traj, s)


    def ramasplit(self):
        print('calculating populations in the Ramachandran space....')
        res_list = sorted([a for a in self.sequence.keys()])
        phi_labs = [el for el in res_list[1:]]
        psi_labs = [el for el in res_list[:-1]]

        self.classdih = dict()
        self.poprama = dict()
        temp_to_ave = dict()
        confs = ['aL', 'aR', 'betaP', 'betaS', 'ND']

        for tt in self.dih.keys():
            self.classdih[tt] = {angle:[] for angle in self.dih[tt].keys()}
            self.poprama[tt] = dict()
            temp_to_ave[tt] = {angle:{c:[] for c in confs} for angle in self.dih[tt].keys()}

        for tt, dihedrals in self.dih.items():
            print(tt+1, '/', len(self.dih.items()))
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
                    self.poprama[tt][res] = dict()
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
                        if phi < 0 and psi > -120 and psi < 50:
                            self.classdih[tt][label_phi].append('aR')
                            self.classdih[tt][label_psi].append('aR')
                            temp_to_ave[tt][label_phi]['aR'].append(phi)
                            temp_to_ave[tt][label_psi]['aR'].append(psi)
                        if phi > -90 and phi < 0 and (psi < -120 or psi > 50):
                            self.classdih[tt][label_phi].append('betaP')
                            self.classdih[tt][label_psi].append('betaP')
                            temp_to_ave[tt][label_phi]['betaP'].append(phi)
                            temp_to_ave[tt][label_psi]['betaP'].append(psi)
                        if phi > -180 and phi < -90 and (psi < -120 or psi > 50):
                            self.classdih[tt][label_phi].append('betaS')
                            self.classdih[tt][label_psi].append('betaS')
                            temp_to_ave[tt][label_phi]['betaS'].append(phi)
                            temp_to_ave[tt][label_psi]['betaS'].append(psi)
                    num_frames = len(these_phi)
                    for c in ['aL', 'aR', 'betaP', 'betaS']:
                        self.poprama[tt][res][c] = len(temp_to_ave[tt][label_phi][c])/num_frames


class dihS2:

    def __init__(self, top, trajs):
        #read sequence
        traj = pt.iterload(trajs[0], top)
        self.sequence = {res.index+1:res.name for res in traj.top.residues}
        self.top = top
        self.trajs = trajs

    def calcS2(self, s='phi', stride=2):
        print('calculating order parameters of '+s+' angles....')
        S2 = dict()

        for tt, t in enumerate(self.trajs):
            traj = pt.iterload(t, self.top, stride=stride)
            #read dihedral angles
            dih = pt.multidihedral(traj, s)
            #get residues
            labels = dih.keys()
            residues = sorted([int(l[len(s)+1:]) for l in labels])
            S2[tt] = dict()
            for r in residues:
                lab = s+':'+str(r)
                vals = np.array(dih[lab])
                S2[tt][r] = (np.sum(np.cos(vals*np.pi/180.))**2+np.sum(np.sin(vals*np.pi/180.))**2)/len(vals)**2
        return S2
