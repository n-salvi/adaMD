import pytraj as pt
from pytraj import vector as va
import numpy as np
from pickle import dump
import sys
sys.path.append('/home/nsalvi')
from adaMD.general import get_today
import matplotlib
matplotlib.use('Agg')
from pylab import *
rcParams['font.family'] = 'FreeSans'
rcParams['legend.numpoints'] = 1
rc('text', usetex=False)
rcParams['mathtext.default'] = 'regular'
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
from adaMD.gaf import mycorrelate2, mycrosscorrelate2
from adaMD.relaxation import unit_vector
from adaMD.fits import smooth_and_fit

from scipy.constants import mu_0, pi, hbar, physical_constants
gamma = physical_constants["proton gyromag. ratio"][0]

searchAT={
'H':'@H', 
'HA':'@HA',
'HB':'@HB',
'HG':'@HG',

'QA':'@HA=', 
'QB':'@HB=',
'QD':'@HD=',
'QG':'@HG=',
'QG2':'@HG=',

'QQD':'@HD=',
'QQG':'@HG=',
}

def define_J(self, amps, taus):
        def J(omega):
            all_terms = [0.4*amp*taus[n]/(1.0+((omega*taus[n])**2)) for n, amp in enumerate(amps)]
            return np.sum(all_terms)
        return J

def getDistances(top, traj, list1, list2):
    #the two lists contain tuples (residue number, atom type)
    #read trajectory
    traj = pt.load(traj, top)
    #read distances
    dist = dict()
    first_in_seq=list(traj.top.residues)[0].index
    for (r1, a1), (r2, a2) in zip(list1, list2):
        dist[str(r1)+a1+':'+str(r2)+a2] = \
            pt.distance(traj, \
            ':'+str(r1-first_in_seq)+searchAT[a1]+' :'+str(r2-first_in_seq)+searchAT[a2])
    dump(dist, open(get_today()+'-rij.pkl', 'wb'))

    #make statistics on distances
    reff = dict()
    for k in sorted(list(dist.keys())):
        temp = np.array(dist[k])**(-6)
        reff[k] = np.mean(temp)**(-1./6)
    dump(reff, open(get_today()+'-reff.pkl', 'wb'))

    #make plots
    with PdfPages(get_today()+'-rij.pdf') as pdf:
        for k in sorted(list(dist.keys())):
            plt.figure()
            plt.hist(dist[k])
            plt.axvline(x=reff[k])
            plt.title(k)
            pdf.savefig()  
            plt.close()

    return [dist, reff]


class NOE:

    def __init__(self, top, traj, list1, list2, rotfit=False, mask='@CA', stride=1):
        #the two lists contain tuples (residue number, atom type)
        #read trajectory
        self.traj = pt.load(traj, top, stride=stride)
        
        self.sequence = {res.index+1:res.name for res in self.traj.top.residues}
        #self.top = top
        #self.traj = traj
        self.n_frames = len(self.traj)
        self.list1 = list1
        self.list2 = list2

    def getDistances(self):
        #read distances
        self.dist = dict()
        first_in_seq=list(self.traj.top.residues)[0].index

        for (r1, a1), (r2, a2) in zip(self.list1, self.list2):
            self.dist[str(r1)+a1+':'+str(r2)+a2] = \
                pt.distance(self.traj, \
                ':'+str(r1-first_in_seq)+searchAT[a1]+' :'+str(r2-first_in_seq)+searchAT[a2])
            #print(str(r1)+a1+':'+str(r2)+a2, self.dist[str(r1)+a1+':'+str(r2)+a2])
        dump(self.dist, open(get_today()+'-rij.pkl', 'wb'))

        #make statistics on distances
        self.reff = dict()
        for k in sorted(list(self.dist.keys())):
            temp = np.array(self.dist[k])**(-6)
            self.reff[k] = np.mean(temp)**(-1./6)
        dump(self.reff, open(get_today()+'-reff.pkl', 'wb'))

        #make plots
        with PdfPages(get_today()+'-rij.pdf') as pdf:
            for k in sorted(list(self.dist.keys())):
                plt.figure()
                plt.hist(self.dist[k])
                plt.axvline(x=self.reff[k])
                plt.title(k)
                pdf.savefig()  
                plt.close()


    def calcACFreff(self):
        self.ACFreff=dict()
        #self.test=dict()
        for (r1, a1), (r2, a2) in zip(self.list1, self.list2):
            label = str(r1)+a1+':'+str(r2)+a2
            first_in_seq=list(self.traj.top.residues)[0].index
            idx1 = pt.select_atoms(searchAT[a1]+ ' & :'+str(r1-first_in_seq), self.traj.top)
            idx2 = pt.select_atoms(searchAT[a2]+ ' & :'+str(r2-first_in_seq), self.traj.top)
            pairs = list(map(list, product(idx1, idx2)))
            data_vec = va.vector_mask(self.traj, pairs, dtype='ndarray')
            if len(data_vec.shape)<3:
                data_vec = [data_vec]
            self.ACFreff[label] = [pt.timecorr(vals, vals, order=2, tstep=1, tcorr=len(vals), norm=False, dtype='ndarray') for vals in data_vec]
            #for testing only: calculate the ACF without pytraj.timecorr
            #self.test[label] = []
            #for vals in data_vec:
                #vals2 = np.array([unit_vector(v) for v in vals])
                #x = vals2[:, 0]
                #y = vals2[:, 1]
                #z = vals2[:, 2]
                #x2 = mycorrelate2(x**2, norm=False)
                #y2 = mycorrelate2(y**2, norm=False)
                #z2 = mycorrelate2(z**2, norm=False)
                #xy = mycorrelate2(x*y, norm=False)
                #yz = mycorrelate2(y*z, norm=False)
                #xz = mycorrelate2(x*z, norm=False)
                #tot = (x2+y2+z2+2*xy+2*xz+2*yz)
                #tot /= tot[0]
                #tot = 1.5*(tot)-.5
            dump(self.ACFreff, open(get_today()+'-ACFreff.pkl', 'wb'))
    
    def calcNOEreff(self, tstep=1.0, numtaus=128):
        print('calculating effective distances\n')
        self.getDistances()
        print('calculating acf of the angular part\n')
        self.calcACFreff()
        del self.traj
        print('fitting correlation functions\n')
        self.ACFreff_fit = dict()
        for label, acfs in self.ACFreff.items():
           acf = np.mean(acfs, axis=0)
           taus, amps, nu1, nu2 = \
               smooth_and_fit(acf, tstep=tstep, lp_threshold=0.15, lp_threshold_2=0.05, mintau=2., numtaus=numtaus)
           self.ACFreff_fit[label] = [taus, amps]
        dump(self.ACFreff_fit, open(get_today()+'-ACFreff-fit.pkl', 'wb'))
        print('calculating NOE rates\n')
        self.sigma_reff = dict()
        for label, fit in self.ACFreff_fit.items():
            taus, amps = fit
            taus = np.array(taus)*1e-12
            sigma = (mu_0/(4*pi))**2
            sigma = sigma*gamma**4*hbar**2*.1
            sigma = sigma/(self.reff[label]*1e-10)**6
            temp = [a*t for a, t in zip(amps, taus)]
            self.sigma_reff[label] = sigma*np.sum(temp)
        dump(self.sigma_reff, open(get_today()+'-sigma-reff.pkl', 'wb')) 
    
    def calcNOEfull(self, tstep=1.0, numtaus=128):
        print('calculating correlation functions\n')
        self.getDistances()
        self.ACFfull=dict()

        for (r1, a1), (r2, a2) in zip(self.list1, self.list2):
            label = str(r1)+a1+':'+str(r2)+a2
            first_in_seq=list(self.traj.top.residues)[0].index
            idx1 = pt.select_atoms(searchAT[a1]+ ' & :'+str(r1-first_in_seq), self.traj.top)
            idx2 = pt.select_atoms(searchAT[a2]+ ' & :'+str(r2-first_in_seq), self.traj.top)
            pairs = list(map(list, product(idx1, idx2)))
            data_vec = va.vector_mask(self.traj, pairs, dtype='ndarray')
            self.ACFfull[label] = []
            if len(data_vec.shape)<3:
                data_vec = [data_vec]
                #self.dist[label] ?
            for n, vals in enumerate(data_vec):
                r = self.dist[label][n]
                cosines = np.array([np.dot(unit_vector(vals[0]), unit_vector(v)) for v in vals])
                cosines = 1.5*cosines**2-.5
                cosines = cosines/((r*1e-10)**3)
                tot = mycorrelate2(cosines, norm=False)
                self.ACFfull[label].append(tot)
        del self.traj
        dump(self.ACFfull, open(get_today()+'-ACFfull.pkl', 'wb'))
        print('fitting correlation functions\n')
        self.ACFfull_fit = dict()
        for label, acfs in self.ACFfull.items():
           #print(acfs)
           acf = np.mean(acfs, axis=0)
           norm_fact = acf[0]
           acf/=acf[0]
           #print(acf)
           taus, amps, nu1, nu2 = \
               smooth_and_fit(acf, tstep=tstep, lp_threshold=0.15, lp_threshold_2=0.05, mintau=2., numtaus=numtaus)
           self.ACFfull_fit[label] = [taus, np.array(amps)*norm_fact]
           #print(taus, np.array(amps)*norm_fact)
        dump(self.ACFfull_fit, open(get_today()+'-ACFfull-fit.pkl', 'wb'))
        print('calculating NOE rates\n')
        self.sigma_full = dict()
        for label, fit in self.ACFfull_fit.items():
            taus, amps = fit
            taus = np.array(taus)*1e-12
            sigma = (mu_0/(4*pi))**2
            sigma = sigma*gamma**4*hbar**2*.1
            #sigma = sigma/(self.reff[label]*1e-10)**6
            temp = [a*t for a, t in zip(amps, taus)]
            self.sigma_full[label] = sigma*np.sum(temp)
        dump(self.sigma_full, open(get_today()+'-sigma-full.pkl', 'wb')) 
        


                
                             
            

        




