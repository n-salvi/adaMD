from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()-2
import numpy as np
from adaMD.fits import smooth_and_fit, TR
from adaMD.relaxation import NMR_relaxation_rates
from adaMD.ga import cxTwoPointCopy
import random
from random import random, seed, randint, uniform
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from pickle import load, dump
from scipy.optimize import leastsq, minimize, basinhopping

def chi2_BH(w, data, amps, taus, B0, thetas, exp_types, use_err=False):
	w = np.abs(w)
	if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
	else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
	out=0

	for nn, dataset in data.items():
		scale = np.std(dataset['val'])
		for rr, res in enumerate(dataset['res']):
				val = dataset['val'][rr]
				err = dataset['err'][rr]
				sims = []
				for n, el in enumerate(w):
					if el>0:
						simulation = NMR_relaxation_rates(amps[n][res], taus[n][res], B0[nn], thetas[res])
						sims.append(simulation.rates[exp_types[nn]]*el)
				sim = np.sum(sims)
				comp = ((sim-val)**2)/scale
				if use_err==True:
					comp /= err
				out += comp
	return out

def deltas(w, data, amps, taus, B0, thetas, exp_types, use_err=False):
	w = np.abs(w)
	if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
	else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
	out=[]

	for nn, dataset in data.items():
		scale = np.std(dataset['val'])
		for rr, res in enumerate(dataset['res']):
				val = dataset['val'][rr]
				err = dataset['err'][rr]
				sims = []
				for n, el in enumerate(w):
					if el>0:
						simulation = NMR_relaxation_rates(amps[n][res], taus[n][res], B0[nn], thetas[res])
						sims.append(simulation.rates[exp_types[nn]]*el)
				sim = np.sum(sims)
				comp = (sim-val)/scale
				if use_err==True:
					comp /= err
				out.append(comp)
	return out


class nofitABSURD:

	def __init__(self, refs, fit_amps, fit_taus, thetas='22'):
		#fit_amps(taus) is a dictionary with keys=segments and the amps(taus) obtained from the fit of the ACFs
		self.amps = load(open(fit_amps, 'rb'))
		self.taus = load(open(fit_taus, 'rb'))
		self.refs = refs
		self.thetas = thetas
		self.resid = sorted(list(self.amps[0].keys()))
		self.num_trajs = len(list(self.amps.keys()))
		print('nofitABSURD object initialised')

	def read_data(self, delta=0):
		print('Reading files')
		self.data = dict()
		self.n_set = len(self.refs)

		for nn, l in enumerate(self.refs):
			stream = np.genfromtxt(l)
			self.data[nn] = dict()
			temp = np.array([int(el)+delta for el in stream[:, 0]])
			self.data[nn]['res'] = temp
			self.data[nn]['val'] = stream[:, 1]
			if len(stream[0]) > 2:
				self.data[nn]['err'] = stream[:, 2]
			else:
				self.data[nn]['err'] = stream[:, 0] - stream[:, 0]

		if self.thetas == '22':
			self.thetas = {res:22 for res in self.resid}
		else:
			temp = np.genfromtxt(self.thetas)
			self.thetas = {int(row[0])+delta:row[1] for row in temp}
			for res in self.resid:
				if res not in self.thetas.keys():
					self.thetas[res] = 22

	def chi2(self, w, use_err=False):
		w = np.abs(w)
		if self.intw==True:
			w = [1 if el > 0.5 else 0 for el in w]
		if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
		else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
		out=0

		for nn, dataset in self.data.items():
			scale = np.std(dataset['val'])
			for rr, res in enumerate(dataset['res']):
				val = dataset['val'][rr]
				err = dataset['err'][rr]
				sims = []
				for n, el in enumerate(w):
					if el>0:
						simulation = NMR_relaxation_rates(self.amps[n][res], self.taus[n][res], self.B0[nn], self.thetas[res])
						sims.append(simulation.rates[self.exp_types[nn]]*el)
				sim = np.sum(sims)
				comp = ((sim-val)**2)/scale
				if use_err==True:
					comp /= err
				out += comp
		return out,

	def chi2step(self, step, use_err=False):
		out=0
		for nn, dataset in self.data.items():
			scale = np.std(dataset['val'])
			for rr, res in enumerate(dataset['res']):
				val = dataset['val'][rr]
				err = dataset['err'][rr]
				sims = []
				for n in range(self.num_trajs):
					simulation = NMR_relaxation_rates(self.amps[n][res], np.array(self.taus[n][res])*step, self.B0[nn], self.thetas[res])
					sims.append(simulation.rates[self.exp_types[nn]])
				sim = np.sum(sims)/self.num_trajs
				comp = ((sim-val)**2)/scale
				if use_err==True:
					comp /= err
				out += comp
		return out

	def chi2_with_step(self, w_with_step, use_err=False):
		fact_step = w_with_step[0]
		w = np.abs(w_with_step[1:])
		if self.intw==True:
			w = [1 if el > 1 else 0 for el in w]
		if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
		else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
		out=0

		for nn, dataset in self.data.items():
			scale = np.std(dataset['val'])
			for rr, res in enumerate(dataset['res']):
				val = dataset['val'][rr]
				err = dataset['err'][rr]
				sims = []
				for n, el in enumerate(w):
					if el>0:
						simulation = NMR_relaxation_rates(self.amps[n][res], np.array(self.taus[n][res])*fact_step, self.B0[nn], self.thetas[res])
						sims.append(simulation.rates[self.exp_types[nn]]*el)
				sim = np.sum(sims)
				comp = ((sim-val)**2)/scale
				if use_err==True:
					comp /= err
				out += comp
		return out,

	def chi2_with_linear_step(self, w_with_step, use_err=False):
		a = np.abs(w_with_step[0])*1000.
		b = np.abs(w_with_step[1])*1000.
		w = np.abs(w_with_step[2:])
		if self.intw==True:
			w = [1 if el > 1 else 0 for el in w]
		if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
		else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
		out=0

		for nn, dataset in self.data.items():
			scale = np.std(dataset['val'])
			for rr, res in enumerate(dataset['res']):
				val = dataset['val'][rr]
				err = dataset['err'][rr]
				sims = []
				for n, el in enumerate(w):
					if el>0:
						corr_taus = np.array([((tau-1)/(a-1))*(b-1)+1 for tau in self.taus[n][res]])
						simulation = NMR_relaxation_rates(self.amps[n][res], corr_taus, self.B0[nn], self.thetas[res])
						sims.append(simulation.rates[self.exp_types[nn]]*el)
				sim = np.sum(sims)
				comp = ((sim-val)**2)/scale
				if use_err==True:
					comp /= err
				out += comp
		return out,

	def step_optimization(self, output='./'):
		out = leastsq(self.chi2step, 1.0, maxfev=5000)
		self.scale = out[0][0]
		for n in range(self.num_trajs):
			for res in self.taus[n].keys():
				self.taus[n][res] = np.array(self.taus[n][res])*self.scale
		with open(output+'time_scale_factor.out', 'w') as tf:
			tf.write(str(self.scale)+'\n')
		with open(output+'scaled_dynamics.out', 'w') as tf:
			residues = sorted(list(self.taus[0].keys()))
			for r in residues:
				for n in range(self.num_trajs):
					for a, t in zip(self.amps[n][r], self.taus[n][r]):
						row = str(r)+' '+str(t)+' '+str(a/self.num_trajs)+'\n'
						tf.write(row)


	def fix_scale(self, scale=1.0, output='./'):
		self.scale = scale
		for n in range(self.num_trajs):
			for res in self.taus[n].keys():
				self.taus[n][res] = np.array(self.taus[n][res])*self.scale
		with open(output+'time_scale_factor.out', 'w') as tf:
			tf.write(str(self.scale)+'\n')
		with open(output+'scaled_dynamics.out', 'w') as tf:
			residues = sorted(list(self.taus[0].keys()))
			for r in residues:
				for n in range(self.num_trajs):
					for a, t in zip(self.amps[n][r], self.taus[n][r]):
						row = str(r)+' '+str(t)+' '+str(a/self.num_trajs)+'\n'
						tf.write(row)


	def optimize(self, intw=True, output='./', ncpu=6, ngen=100, popsize=150):

		self.intw=intw
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
		toolbox = base.Toolbox()
		pool = multiprocessing.Pool(ncpu)
		toolbox.register("map", pool.map)
		toolbox.register("attr_float", random)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len([k for k in self.amps.keys()]))
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.chi2)
		toolbox.register("mate", cxTwoPointCopy)
		toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
		toolbox.register("select", tools.selTournament, tournsize=3)
		seed()
		pop = toolbox.population(n=popsize)

		hof = tools.HallOfFame(1, similar=np.array_equal)

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		print('Starting GA')
		algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof)

		result = hof[0]

		if intw == True:
			self.result = [1 if el > 0.5 else 0 for el in result]
		else:
			self.result = np.abs(result)

		np.savetxt(output+'final_weights.out', self.result)

	def optimize_leastsq(self, output='./'):
		nvars=len([k for k in self.amps.keys()])
		x0 = [1 for el in range(nvars)]

		result = leastsq(deltas, x0, args=(self.data, self.amps, self.taus, self.B0, self.thetas, self.exp_types, False,)) #, maxfev=100000)
		print(result)
		w = np.abs(result[0])
		if np.sum(w)>0:
				w = np.array(w)/float(np.sum(w))
		else:
				w = [1 for el in w]
				w = np.array(w)/float(np.sum(w))
		self.result = w

		np.savetxt(output+'final_weights.out', self.result)


	def optimize_with_step(self, intw=True, output='./', ngen=50, popsize=150, ncpu=6):

		self.intw=intw
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
		toolbox = base.Toolbox()
		pool = multiprocessing.Pool(ncpu)
		toolbox.register("map", pool.map)
		toolbox.register("attr_float", uniform, 0.1, 1.9)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len([k for k in self.amps.keys()])+1)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.chi2_with_step)
		toolbox.register("mate", cxTwoPointCopy)
		toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
		toolbox.register("select", tools.selTournament, tournsize=3)
		seed()
		pop = toolbox.population(n=popsize)

		hof = tools.HallOfFame(1, similar=np.array_equal)

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		print('Starting GA')
		algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof)

		result = hof[0]

		if intw == True:
			self.result = [1 if el > 1 else 0 for el in result[1:]]
		else:
			self.result = np.abs(result[1:])

		self.result_scale_fact = result[0]
		np.savetxt(output+'final_weights.out', self.result)
		with open(output+'time_scale_factor.out', 'w') as tf:
			tf.write(str(self.result_scale_fact)+'\n')

	def optimize_with_linear_step(self, intw=True, output='./', ngen=50, popsize=150, ncpu=6):

		self.intw=intw
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
		toolbox = base.Toolbox()
		pool = multiprocessing.Pool(ncpu)
		toolbox.register("map", pool.map)
		toolbox.register("attr_float", uniform, 0.1, 1.9)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len([k for k in self.amps.keys()])+2)
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.chi2_with_linear_step)
		toolbox.register("mate", cxTwoPointCopy)
		toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
		toolbox.register("select", tools.selTournament, tournsize=3)
		seed()
		pop = toolbox.population(n=popsize)

		hof = tools.HallOfFame(1, similar=np.array_equal)

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		print('Starting GA')
		algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof)

		result = hof[0]

		if intw == True:
			self.result = [1 if el > 1 else 0 for el in result[2:]]
		else:
			self.result = np.abs(result[2:])

		self.result_scale_fact = np.array(np.abs(result[:2]))*1000.
		self.result_a = np.abs(result[0])*1000.
		self.result_b = np.abs(result[1])*1000.
		np.savetxt(output+'final_weights.out', self.result)
		with open(output+'time_scale_factor.out', 'w') as tf:
			tf.write(str(self.result_scale_fact)+'\n')


	def optimal_dynamics(self, output='./'):
		result = np.array(self.result)/np.sum(self.result)
		residues = sorted(list(self.amps[0].keys()))
		with open(output+'dynamics.out', 'w') as tf:
			for r in residues:
				for n, el in enumerate(result):
					if el > 0:
						for a, t in zip(self.amps[n][r], self.taus[n][r]):
							row = str(r)+' '+str(t)+' '+str(a*el)+'\n'
							tf.write(row)


	def optimal_dynamics_with_step(self, output='./'):
		result = np.array(self.result)/np.sum(self.result)
		residues = sorted(list(self.amps[0].keys()))
		with open(output+'dynamics.out', 'w') as tf:
			for r in residues:
				for n, el in enumerate(result):
					if el > 0:
						for a, t in zip(self.amps[n][r], self.taus[n][r]):
							row = str(r)+' '+str(t*self.result_scale_fact)+' '+str(a*el)+'\n'
							tf.write(row)

	def optimal_dynamics_with_linear_step(self, output='./'):
		result = np.array(self.result)/np.sum(self.result)
		residues = sorted(list(self.amps[0].keys()))
		with open(output+'dynamics.out', 'w') as tf:
			for r in residues:
				for n, el in enumerate(result):
					if el > 0:
						for a, t in zip(self.amps[n][r], self.taus[n][r]):
							row = str(r)+' '+str(((t-1)/(self.result_a-1))*(self.result_b-1)+1)+' '+str(a*el)+'\n'
							tf.write(row)


class ABSURD:

	def __init__(self, refs, acf_instance, tstep, thetas='22', rotfit=False):
		#acf_instance here is a result of a run of adaMD.relaxation.nh_acf
		self.acf = acf_instance
		self.refs = refs
		self.thetas = thetas
		self.tstep = tstep
		self.skip = 2
		print('ABSURD object initialised')
		self.resid = sorted([k for k in self.acf.acf[0].keys()])

	def read_data(self, delta=0):
		print('Reading files')
		self.data = dict()
		self.n_set = len(self.refs)

		for nn, l in enumerate(self.refs):
			stream = np.genfromtxt(l)
			self.data[nn] = dict()
			temp = np.array([int(el)+delta for el in stream[:, 0]])
			self.data[nn]['res'] = temp
			self.data[nn]['val'] = stream[:, 1]
			if len(stream[0]) > 2:
				self.data[nn]['err'] = stream[:, 2]
			else:
				self.data[nn]['err'] = stream[:, 0] - stream[:, 0]

		if self.thetas == '22':
			self.thetas = {res:22 for res in self.resid}
		else:
			temp = np.genfromtxt(self.thetas)
			self.thetas = {int(row[0])+delta:row[1] for row in temp}
			for res in self.resid:
				if res not in self.thetas.keys():
					self.thetas[res] = 22

	def acfOne(self, res, w):
		lengths = [len(self.acf.acf[n][res]) for n, val in enumerate(w)]
		min_length = min(lengths)
		all_acf = [val*self.acf.acf[n][res][:min_length:self.skip] for n, val in enumerate(w) if val > 0]
		ave_acf = np.sum(all_acf, axis=0)/float(np.sum(w))
		#tau_estimates, amp_estimates, simulated, x = smooth_and_fit(ave_acf,
			#tstep=self.tstep, lp_threshold=0.15, lp_threshold_2=0.05, mintau=self.tstep, numtaus=1024)

		#tau_estimates, amp_estimates, simulated, x = TR(ave_acf,
			#tstep=self.tstep*self.skip, mintau=self.tstep, maxtau=len(ave_acf)*self.tstep*self.skip/2., numtaus=128)

		return ave_acf

	def valuesOne(self, res, w):
		lengths = [len(self.acf.acf[n][res]) for n, val in enumerate(w)]
		min_length = min(lengths)
		all_acf = [val*self.acf.acf[n][res][:min_length:self.skip] for n, val in enumerate(w) if val > 0]
		ave_acf = np.sum(all_acf, axis=0)/float(np.sum(w))
		tau_estimates, amp_estimates, simulated, x = smooth_and_fit(ave_acf,
			tstep=self.tstep, lp_threshold=0.15, lp_threshold_2=0.05, mintau=self.tstep, numtaus=1024)

		#tau_estimates, amp_estimates, simulated, x = TR(ave_acf,
			#tstep=self.tstep*self.skip, mintau=self.tstep, maxtau=len(ave_acf)*self.tstep*self.skip/2., numtaus=128)

		sims = [NMR_relaxation_rates(amp_estimates, tau_estimates, B0, self.thetas[res]) for B0 in self.B0]

		values = {'exp':[], 'sim':[]}
		for n, k in enumerate(list(self.data.keys())):
			if res in self.data[k]['res']:
				pos = list(self.data[k]['res']).index(res)
				values['exp'].append(self.data[k]['val'][pos])
				values['sim'].append(sims[k].rates[self.exp_types[k]])
		return values

	def simOne(self, res, w):
		lengths = [len(self.acf.acf[n][res]) for n, val in enumerate(w)]
		min_length = min(lengths)
		all_acf = [val*self.acf.acf[n][res][:min_length:self.skip] for n, val in enumerate(w) if val > 0]
		ave_acf = np.sum(all_acf, axis=0)/float(np.sum(w))
		tau_estimates, amp_estimates, simulated, x = smooth_and_fit(ave_acf,
			tstep=self.tstep, lp_threshold=0.15, lp_threshold_2=0.05, mintau=self.tstep, numtaus=8)

		# tau_estimates, amp_estimates, simulated, x = TR(ave_acf,
		# 	tstep=self.tstep*self.skip, mintau=self.tstep, maxtau=len(ave_acf)*self.tstep*self.skip/2., numtaus=32)

		sims = [NMR_relaxation_rates(amp_estimates, tau_estimates, B0, self.thetas[res]) for B0 in self.B0]

		chi2 = []
		for n, k in enumerate(list(self.data.keys())):
			if res in self.data[k]['res']:
				pos = list(self.data[k]['res']).index(res)
				if np.sum(self.data[k]['err']) > 0:
					d = (self.data[k]['val'][pos]-sims[k].rates[self.exp_types[k]])/np.std(self.data[k]['val'])# self.data[k]['err'][pos]
					#print(self.data[k]['val'][pos], sims[k].rates[self.exp_types[k]])
				else:
					d = (self.data[k]['val'][pos]-sims[k].rates[self.exp_types[k]])/np.std(self.data[k]['val'])
				chi2.append(d**2)

		return np.sum(chi2)

	def chi2(self, w):
		if self.intw==True:
			w = [1 if el > 0.5 else 0 for el in w]
		if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
		else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
		#fit all weighted ACFs and calculates chi2
		#sims = Parallel(n_jobs=num_cores)(delayed(self.simOne)(res=res, w=w) for res in self.resid)
		sims = [self.simOne(res=res, w=w) for res in self.resid]

		return np.sum(sims),

	def optimalvalues(self, w):
		if self.intw==True:
			w = [1 if el > 0.5 else 0 for el in w]
		if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
		else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
		values = {res:self.valuesOne(res=res, w=w) for res in self.resid}

		return values


	def optimalacf(self, w):
		if self.intw==True:
			w = [1 if el > 0.5 else 0 for el in w]
		if np.sum(w)>0:
			w = np.array(w)/float(np.sum(w))
		else:
			w = [1 for el in w]
			w = np.array(w)/float(np.sum(w))
		values = {res:self.acfOne(res=res, w=w) for res in self.resid}

		return values


	def optimize(self, intw=True, output='./', npop=150, ngen=100):

		self.intw=intw
		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)
		toolbox = base.Toolbox()
		toolbox.register("attr_float", random)
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len([k for k in self.acf.acf.keys()]))
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("evaluate", self.chi2)
		toolbox.register("mate", cxTwoPointCopy)
		toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
		toolbox.register("select", tools.selTournament, tournsize=3)
		seed()
		pop = toolbox.population(n=npop)

		hof = tools.HallOfFame(1, similar=np.array_equal)

		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.mean)
		stats.register("std", np.std)
		stats.register("min", np.min)
		stats.register("max", np.max)

		print('Starting GA')
		algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=ngen, stats=stats, halloffame=hof)

		result = hof[0]

		if intw == True:
			self.result = [1 if el > 0.5 else 0 for el in result]
		else:
			self.result = result

		np.savetxt(output+'final_weights.out', self.result)
