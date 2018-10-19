import numpy as np
import random

tolerance = 0.001
sparsity_weight = 0.05
coherence_weight = 0.075

'''
Little script that tests different coherence metrics on fake rationales. Used to play around with alternatives to group lasso used in original paper. 
'''

def main():



	
	zd1 = [0,0,1,0,0,0,0,1,0,0]
	zd2 = [0,1,1,0,0,0,0,1,1,0]
	zd3 = [0,1,1,1,0,0,1,1,1,0]
	zd4 = [0,0,0,0,0,0,0,0,0,0]
	zd5 = [1,1,1,1,1,1,1,1,1,1]
	zd6 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	zd7 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	zd8 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
	zd9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	zd10 = [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
	zd11 = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
	zd12 = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
	zd13 = [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]

	zd14 = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
	zd15 = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

	zd16 = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
	zd17 = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]

	
	zc1 = [.1,.1,.9,.1,.1,.1,.1,.9,.1,.1]
	zc2 = [.1,.9,.9,.1,.1,.1,.1,.9,.9,.1]
	zc3 = [.1,.9,.9,.9,.1,.1,.9,.9,.9,.1]
	zc6 = [.1,.9,.6,.9,.1,.1,.9,.6,.9,.1]
	zc4 = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
	zc5 = [.2, .8, .8, .2, .2, .2, .2, .8, .8, .2]


	discrete = [np.asarray(z) for z in [zd1,zd2,zd3,zd4,zd5,zd6,zd7,zd8,zd9, zd10, zd11, zd12,zd13,zd14,zd15,zd16,zd17]]
	continuous = [np.asarray(z) for z in [zc1,zc2,zc5,zc3,zc6,zc4]]

	# functions = [pz, zdiff, zdiff_mean, multiplicative_zdiff, pneq, p0g1, p1g0, combined_2, continuous_p0g1, multiplicative_p0g1, combined]
	functions = [pz, zdiff_mean, multiplicative_zdiff, pz_mult_zdiff]

	print "Discrete zs:"
	run_functions_over_each_z(discrete, functions)

	print "Continuous zs:"
	run_functions_over_each_z(continuous, functions)

	# print "Random zs:"
	# random.seed(92277)
	# random_zs = np.round(np.random.rand(5,10),2)
	# run_functions(random_zs, functions)


	tests = [test1, test2, test3, test4,test5,test6]
	print 'Evaluating functions:'
	evaluate_each_function(functions, zs = discrete+continuous, tests=tests)


def run_functions_over_each_z(zs, functions):
	for z in zs:
		print '\tz   :',z
		print '\tzt-1:',z[:-1]
		print '\tzt  :', z[1:]

		for function in functions:
			name, value = function(z)
			print '\t\t{}: {}'.format(name, value)
		print

def evaluate_each_function(functions, zs = [], tests = []):
	for function in functions:
		name = function(np.asarray([0,1,0]))[0]
		print '\t{}'.format(name)
		print '\tzs:'
		for z in zs:
			value = function(z)[1]
			print '\t\t{}: {}'.format(z,value)
		print '\tTests:'
		for test in tests:
			description,outcome = test(function)
			indented_description = '\t\t'+description.replace('\n','\n\t\t')
			print indented_description
			if outcome:
				print '\t\t\tSUCCEEDED'
			else:
				print '\t\t\tFAILED'
			print

			
			
	

def pz(z):
	return 'p(z) (aka L1 norm)',np.mean(z)

def zdiff(z):
	return 'zdiff sum',np.sum(np.abs(z[:-1] - z[1:]))

def pneq(z):
	t_eq_tm1 = np.sum(z[:-1] == z[1:])
	t_neq_tm1 = np.sum(z[:-1] != z[1:])
	return 'p(zt != zt-1)', t_neq_tm1 / (t_eq_tm1 + t_neq_tm1 + 0.0001)

def zdiff_mean(z):
	return 'zdiff mean', np.mean(np.abs(z[:-1] - z[1:]))

def multiplicative_zdiff(z):
	return 'multiplicative zdiff', np.mean((1-z[:-1]) * z[1:]) + np.mean(z[:-1] * (1-z[1:]))

def alternateie_multiplicative_zdiff(z):
	return 'multiplicative zdiff', np.mean((1-z[:-1]) * z[1:]) + np.mean(z[:-1] * (1-z[1:]))

def pz_mult_zdiff(z):
	return '{}*pz + {}*multiplicative zdiff'.format(sparsity_weight, coherence_weight),sparsity_weight*pz(z)[1]+coherence_weight*multiplicative_zdiff(z)[1]

def normalized_multiplicative_zdiff(z):
	val = 1- (np.mean(z[:-1] * z[1:]) / (np.mean(z) + 0.00001))

	return 'normalized multiplicative zdiff',val

def p0g1(z):
	teq1_tm1eq1 = np.sum(np.logical_and(z[:-1], z[1:]))
	teq0_tm1eq1 =  np.sum(np.logical_and(z[:-1],1-z[1:]))
	rp0g1 = teq0_tm1eq1/(teq1_tm1eq1+teq0_tm1eq1+0.0001)
	return 'P(zt=0|zt-1=1)',rp0g1


def p1g0(z):
	teq0_tm1eq0 = np.sum(1-np.logical_and(z[:-1],z[1:]))
	teq1_tm1eq0 = np.sum(np.logical_and(z[:-1],1-z[1:]))
	rp1g0 = teq1_tm1eq0 / (teq0_tm1eq0 + teq1_tm1eq0 + 0.0001)
	return 'P(zt=1|zt-1=0)', rp1g0


def combined_2(z):
	return 'combined p0g1 and p1g0',(p1g0(z)[1]+p0g1(z)[1])/2

def continuous_p0g1(z):
	return 'additive continuous P(zt=0|zt-1=1)',np.sum((z[:-1] - z[1:])*z[:-1])/float(np.sum(z[:-1]+0.0001))

def multiplicative_p0g1(z):
	teq1_tm1eq1 = tmeq1_teq1 = np.sum(z[:-1] * z[1:])

	teq0_tm1eq1 =  np.sum((1-z[:-1])*z[1:])
	tm1eq0_teq1 =  np.sum(z[:-1]*(1-z[1:]))

	p0g1 = ((teq0_tm1eq1)/((teq0_tm1eq1+teq1_tm1eq1+0.0001)) + (tm1eq0_teq1)/((tm1eq0_teq1+tmeq1_teq1+0.0001)))/2
	return 'multiplicative P(zt=0|zt-1=1) and P(zt-1=0|zt=1)',p0g1

def l2_norm(z):
	return 'L2 norm',np.mean(z ** 2)

def l_half_norm(z):
	return 'L-0.5 norm',np.mean(z ** 0.5)

def e_norm(z):
	return 'e norm',np.mean(np.e ** z)

def combined(z):

	return 'combined multiplicative p1g0 and pz', 0.025 * multiplicative_p0g1(z)[1] + 0.05 * pz(z)[1]

def gini(z):
	return 'gini impurity',np.mean(z*(1-z))

def test1(func):
	z1 = np.asarray([0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
	z2 = np.asarray([0,0,1,0,0,0,0,1,0,0])

	description = "Long strings of 1s:\n{}({:.3f})\n\tLess than\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])

	return description, (func(z2)[1] - func(z1)[1]) > tolerance

def test2(func):
	z1 = np.asarray([0, 0,0,1,1,1,1,0,0, 0])
	z2 = np.asarray([0,0,0,1,0,0,1,0,0,0])

	description = "Fill in gaps:\n{}({:.3f})\n\tLess than\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])

	return description, (func(z2)[1] - func(z1)[1]) > tolerance

def test3(func):
	z1 = np.asarray([0,0,0,1,1,1,1,1,0,0])
	z2 = np.asarray([0,1, 0, 1, 0, 1, 0, 1, 0, 0])

	description = "Consolidate 1s:\n{}({:.3f})\n\tLess than\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])

	return description, (func(z2)[1] - func(z1)[1]) > tolerance

def test4(func):
	z1 = np.asarray([.1,.9,.9,.1,.1,.1,.9,.9,.1,.1])
	z2 = np.asarray([.2, .8, .8, .2, .2, .2, .2, .8, .8, .2])

	description = "Zs closer to 1s and 0s:\n{}({:.3f})\n\tLess than\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])

	return description, (func(z2)[1] - func(z1)[1]) > tolerance


def test5(func):


	z1 = np.asarray([.1,.9,.9,.9,.1,.1,.9,.9,.9,.1])
	z2 = np.asarray([.1,.9,.6,.9,.1,.1,.9,.6,.9,.1])

	description = "Fill in gaps with 1s or 0s:\n{}({:.3f})\n\tLess than\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])

	return description, (func(z2)[1] - func(z1)[1]) > tolerance

def test6(func):
	z1 = np.asarray([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	z2 = np.asarray([0,0, 0, 0, 0, 0, 0, 0, 0, 1])

	description = "Symmetric edge cases:\n{}({:.3f})\n\tEqual to\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])

	return description, (func(z2)[1] == func(z1)[1])


def test7(func):
	z1 = np.asarray([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
	z2 = np.asarray([0,1, 1, 0, 0, 0, 0, 0, 0, 0])

	description = "Edges not treated differently from interior:\n{}({:.3f})\n\tEqual to\n{}({:.3f})".format(z1,func(z1)[1],z2,func(z2)[1])
	return description, (func(z2)[1] == func(z1)[1])


if __name__ == "__main__":
	main()