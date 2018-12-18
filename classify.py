import pytraj as pt
from pytraj import vector as va
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use('pdf')
from pylab import *
#from matplotlib import *
#import matplotlib.pyplot as plt


from matplotlib.ticker import MultipleLocator, AutoMinorLocator

from pickle import dump

from adaMD.ga import cxTwoPointCopy
import random
from random import random, seed, randint, uniform
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def distance(v1, v2):
	return np.sqrt(np.sum((v1-v2)**2))

def contact_map(positions):
	return np.array([[distance(v1, v2) for v2 in positions] for v1 in positions])

def calc_rg(traj):
	return pt.radgyr(traj, '@CA')

def calc_dih(traj):
	return pt.multidihedral(traj, 'phi psi')

def calc_sigmas(traj):
	#read vectors
	ca_indices = pt.select_atoms('@CA', traj.top)
	n_indices = pt.select_atoms('@N', traj.top)
	can_pairs = np.array(list(zip(ca_indices[1:], n_indices[1:])))
	data_vec = va.vector_mask(traj, can_pairs, dtype='ndarray')
	#FINISH THIS

calc_values = dict()
calc_values['rg'] = calc_rg
calc_values['bb'] = calc_dih
calc_values['seg'] = calc_sigmas

class convergence2:

	def __init__(self, top, ens, stride=1, properties=['rg', 'bb']):
		traj = pt.iterload(ens, top, stride=stride)
		self.properties = properties
		self.ens_values = {p:calc_values[p](traj) for p in properties}
		print('Properties of the ensemble calculated')


class convergence:

	def __init__(self, top, ens, stride=1):
		traj = pt.iterload(ens, top, stride=stride)
		self.dih_ens = pt.multidihedral(traj, 'phi psi')
		self.rg_ens = pt.radgyr(traj, '@CA')

	def score(self, w):
		if self.starting:
			for n in self.start_points:
				w[n-1]=1.1
		ind = np.argpartition(w, -self.size)[-self.size:]
		w_dih = {k:[self.dih_ens[k][n] for n in ind] for k in self.dih_ens.keys()}
		w_rg = [self.rg_ens[n] for n in ind]
		D = dict()
		for k in self.dih_ens.keys():
			D[k], not_used = stats.ks_2samp(self.dih_ens[k], w_dih[k])
		D['rg'] = stats.ks_2samp(self.rg_ens, w_rg)[0]*len(list(self.dih_ens.keys()))
		return np.sum(list(D.values())),


	def select_seeds(self, size=20, output='./', starting=False, startfile='calculated.in'):
		if starting:
			self.starting=True
			start_points = np.genfromtxt(startfile)
			self.start_points = [int(el) for el in start_points]
			if len(self.start_points) > size-1:
				raise ValueError("Required ensemble size has to be larger than number of trajectories already calculated.")
			else:
				self.size=size
		else:
			self.starting=False
			self.start_points = []
			self.size=size
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
		toolbox = base.Toolbox()
		toolbox.register("attr_float", random)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(self.rg_ens))
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.score)
		toolbox.register("mate", cxTwoPointCopy)
		toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
		toolbox.register("select", tools.selTournament, tournsize=3)
		seed()
		pop = toolbox.population(n=150)

		hof = tools.HallOfFame(1, similar=np.array_equal)

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		print('Starting GA')
		algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof)

		result = hof[0]
		if self.starting:
			for n in self.start_points:
				result[n-1]=1.1
		self.result = np.argpartition(result, -self.size)[-self.size:]

		with open(output+'selected_seeds.out', 'w') as tf:
			for el in self.result:
				if int(el+1) not in self.start_points:
					tf.write(str(int(el+1))+'\n')



	def calc_values(self, top, trajs):
		traj_obj = {t:pt.iterload(t, top) for t in trajs}
		#lengths = [len(traj_obj[t]) for t in trajs]
		#stride = int(np.sum(lengths)/len(self.rg_ens))
		#traj_obj = {t:pt.iterload(t, top, frame_slice=(0, -1, stride)) for t in trajs}
		dih_values = {t:pt.multidihedral(tt, 'phi psi') for t, tt in traj_obj.items()}
		rg_values = {t:pt.radgyr(tt, '@CA') for t, tt in traj_obj.items()}
		self.dih_sim = dict()
		for k in self.dih_ens.keys():
			for t, valset in dih_values.items():
				if k in self.dih_sim.keys():
					self.dih_sim[k] = self.dih_sim[k] + list(valset[k])
				else:
					self.dih_sim[k] = list(valset[k])
		self.rg_sim = []
		for t, valset in rg_values.items():
			self.rg_sim += list(valset)


	def calc_binned_values(self, top, trajs):
		###CHANGE THIS
		traj_obj = {t:pt.iterload(t, top) for t in trajs}
		dih_values = {t:pt.multidihedral(tt, 'phi psi') for t, tt in traj_obj.items()}
		rg_values = {t:pt.radgyr(tt, '@CA') for t, tt in traj_obj.items()}
		self.dih_sim = dict()
		for k in self.dih_ens.keys():
			for t, valset in dih_values.items():
				if k in self.dih_sim.keys():
					self.dih_sim[k] = self.dih_sim[k] + list(valset[k])
				else:
					self.dih_sim[k] = list(valset[k])
		self.rg_sim = []
		for t, valset in rg_values.items():
			self.rg_sim += list(valset)

	def perform_KS(self, alpha=0.1):
		self.D = dict()
		self.pvalues = dict()
		for k in self.dih_ens.keys():
			self.D[k], self.pvalues[k] = stats.ks_2samp(self.dih_ens[k], self.dih_sim[k])
		self.D['rg'], self.pvalues['rg'] = stats.ks_2samp(self.rg_ens, self.rg_sim)
		x_phi = []
		y_phi = []
		x_psi = []
		y_psi = []
		for k, val in self.D.items():
			if 'phi' in k:
				x_phi.append(int(k[4:]))
				y_phi.append(val)
			if 'psi' in k:
				x_psi.append(int(k[4:]))
				y_psi.append(val)
		min_res = min([min(x_phi), min(x_psi)])
		max_res = max([max(x_phi), max(x_psi)])
		x_phi = []
		y_phi = []
		x_psi = []
		y_psi = []
		for k, val in self.D.items():
			if 'phi' in k:
				if int(k[4:])<max_res:
					x_phi.append(int(k[4:]))
					y_phi.append(val)
			if 'psi' in k:
				if int(k[4:])>min_res:
					x_psi.append(int(k[4:]))
					y_psi.append(val)

		c = np.sqrt(-.5*np.log(alpha/2.))
		n = len(self.rg_ens)
		m = len(self.rg_sim)
		D_critical = c*np.sqrt((n+m)/(n*m))
		rcParams['font.family'] = 'FreeSans'
		rcParams['legend.numpoints'] = 1
		rc('text', usetex=False)
		rcParams['mathtext.default'] = 'regular'
		tableau10B = [(0, 107,164), (255,128,14), (171, 171, 171),\
		(89, 89, 89), (95, 158, 209), (200, 82, 0), (137, 137, 137),\
		(162, 200, 236), (255,188,121), (207,207,207)]
		for i in range(len(tableau10B)):
			r, g, b = tableau10B[i]
			tableau10B[i] = (r / 255., g / 255., b / 255.)

		f = plt.figure(figsize=(8, 8))
		ax = f.add_subplot(711)
		plt.ylabel(r'$D_{ens, MD}$', fontsize=14)
		plt.xlabel(r'Sequence', fontsize=14)
		plt.bar(np.array(x_phi)-0.5, y_phi, 1., color=tableau10B[0], lw=0, alpha=0.7)
		plt.bar(np.array(x_psi)-0.5, y_psi, 1., color=tableau10B[1], lw=0, alpha=0.7)
		plt.bar([max_res+1.5], [self.D['rg']], 2., color=tableau10B[2], lw=0, alpha=0.7)
		x = [el for el in range(min_res-1, max_res+3)]
		y = [D_critical for el in x]
		plt.plot(x, y, lw=2, color=tableau10B[3])
		plt.xticks(range(max_res+6)[0::10])
		plt.xlim([min_res-1, max_res+2])
		locator_params(axis='y', nbins=5)
		minor_locator = AutoMinorLocator(2)
		ax.yaxis.set_minor_locator(minor_locator)
		minorX = MultipleLocator(5)
		ax.xaxis.set_minor_locator(minorX)
		plt.savefig('convergence.pdf', dpi=300, transparent=True, orientation='landscape')

		with open('convergence.pkl', 'wb') as tf:
			dump([self.D, self.pvalues, D_critical], tf)


class classify:

	def __init__(self, pdbs):
		self.pdbs = pdbs
		self.ens_size = len(pdbs)
		self.ens = [pt.load(f) for f in pdbs]
		self.ens_values = [contact_map(f['@CA'].xyz[0]) for f in self.ens]
		self.ens_diffs = np.array([[np.linalg.norm(val1-val2, ord='fro') for val2 in self.ens_values] for val1 in self.ens_values] )
		mask = np.ones((self.ens_size,self.ens_size))
		self.mask = (mask - np.diag(np.ones(self.ens_size))).astype(np.bool)
		self.mean_dist = np.mean(self.ens_diffs[self.mask])
		self.min_dist = np.amin(self.ens_diffs[self.mask])
		self.std_dist = np.std(self.ens_diffs[self.mask])
		print('ensemble loaded')

	def load_trajectories(self, top, trajs, stride=1):
		self.num_trajs = len(trajs)
		self.trajs = {t:pt.iterload(t, top, frame_slice=(0, -1, stride)) for t in trajs}
		self.traj_values = {t:[contact_map(f) for f in self.trajs[t]['@CA']] for t in self.trajs.keys()}
		print('trajectories loaded')

	def classify(self):
		self.result = {t:[] for t in self.traj_values.keys()}
		self.values = {t:[] for t in self.traj_values.keys()}
		self.warn_count = dict()
		for t, values in self.traj_values.items():
			warn_count = 0
			for val1 in values:
				diffs = [np.linalg.norm(val1-val2, ord='fro') for val2 in self.ens_values]
				opt_val = min(diffs)
				if opt_val < self.mean_dist:
				#if opt_val < self.min_dist:
					selected = diffs.index(opt_val)
					self.result[t].append(selected)
				else:
					warn_count += 1
				self.values[t].append(opt_val)
			self.warn_count[t] = warn_count
		print('finished classifying')

	def count(self):
		self.counts = {n:0 for n in range(self.ens_size)}
		for t, results in self.result.items():
			for el in results:
				self.counts[el] = self.counts[el]+1
		print('finished counting')

	def pick_structure(self):
		sk = sorted([k for k in self.counts.keys()])
		svals = [self.counts[k] for k in sk]
		not_sampled = [n for n, val in enumerate(svals) if val==0]
		if len(not_sampled)>0:
			print(str(len(not_sampled))+' conformations not sampled.')
			if len(not_sampled) <= 4:
				to_print = [self.pdbs[v] for v in not_sampled]
			else:
				new_runs = np.random.choice(not_sampled, 4, replace=False)
				to_print = [self.pdbs[v] for v in new_runs]
			print('Suggested starting point for next trajectories:', to_print)
		else:
			print('All conformations were sampled.')
		max_count = max(svals)
		pos_max = svals.index(max_count)
		print('Most visited structure:', self.pdbs[pos_max], '( count =', str(int(max_count)), ')')
		median_count = np.median(svals)
		print('Median count:', str(median_count))


class contacts:

	def __init__(self, top, trajs, stride=1):
		self.top = top
		self.num_trajs = len(trajs)
		self.trajs = {t:pt.iterload(t, top, frame_slice=(0, -1, stride)) for t in trajs}
		self.traj_values = {t:np.mean([contact_map(f) for f in self.trajs[t]['@CA']], axis=0) for t in self.trajs.keys()}
		self.ave_map = np.mean([self.traj_values[t] for t in self.trajs.keys()], axis=0)
		print('calculation done')
