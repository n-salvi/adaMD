import numpy as np
import random
from random import random, seed, randint
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

def cxTwoPointCopy(ind1, ind2):
	"""Execute a two points crossover with copy on the input individuals. The
	copy is required because the slicing in numpy returns a view of the data,
	which leads to a self overwritting in the swap operation. It prevents
	::

		>>> import numpy
		>>> a = numpy.array((1,2,3,4))
		>>> b = numpy.array((5.6.7.8))
		>>> a[1:3], b[1:3] = b[1:3], a[1:3]
		>>> print(a)
		[1 6 7 4]
		>>> print(b)
		[5 6 7 8]
	"""
	size = len(ind1)
	cxpoint1 = randint(1, size)
	cxpoint2 = randint(1, size - 1)
	if cxpoint2 >= cxpoint1:
		cxpoint2 += 1
	else:  # Swap the two cx points
		cxpoint1, cxpoint2 = cxpoint2, cxpoint1

	ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
		= ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

	return ind1, ind2
