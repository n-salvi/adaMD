import pytraj as pt
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

searchAT={
'H':'@H', 'QD':'@HD=',
'QA':'@HA=', 'QB':'@HB=',
'HB':'@HB', 'QG':'@HG=',
'HA':'@HA'
}

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
            ':'+str(r1+2-first_in_seq)+searchAT[a1]+' :'+str(r2+2-first_in_seq)+searchAT[a2])
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
