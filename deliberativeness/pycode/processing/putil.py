from datetime import datetime, time
import nltk
import numpy
import numpy as np
import json
import pandas
import itertools

import pytz

from bs4 import BeautifulSoup
from copy import copy
import intervaltree as it
import sys
import os
import re
from scipy import stats

from collections import OrderedDict
import math
import traceback
# import smtplib
# from email.mime.text import MIMEText

'''
A giant undifferentiated grab-bag of utility functions
'''

#Columns that should be included in a processed data file
unlabeled_data_columns = ["text", "datetime", "platform_id", "platform_comment_id", "url", "original_text", "tokenization"]

labeled_data_columns = ["text", "datetime", "platform_id", "platform_comment_id", "url", "original_text", "tokenization", 'target']

synthetic_data_columns = ["text", "datetime", "platform_id", "platform_comment_id", "url", "original_text", "tokenization", 'target', 'rationale']

def year2datetime(year):
	return datetime(year=year, month=1, day=1)

def replace_wiki_tokens(s):
	'''
	Replace tokens specific to the Wikipedia data with correct characters
	:param s:
	:return:
	'''
	return s.replace('TAB_TOKEN','\t').replace('NEWLINE_TOKEN','\n')

def replace_wiki_tokens_with_placeholders(s):
	'''
	Replace tokens specific to the Wikipedia data with correct characters
	:param s:
	:return:
	'''
	return s.replace('TAB_TOKEN',' ').replace('NEWLINE_TOKEN',' ').strip()



tokenizer = nltk.tokenize.WordPunctTokenizer()

def process_text_to_pd(s, unicode = False):

	return pandas.Series(process_text_to_strings(s, unicode))

def process_text_to_strings(s, unicode = False, ptokenizer = tokenizer):
	'''
	Process a text for modeling. Return both a processed version of the text and a description
	of how it was tokenized.
	:param s:
	:return:
	'''

	tokens, spans= process_text(s, unicode,ptokenizer=ptokenizer)

	span_text = json.dumps(spans)
	processed_text = ' '.join(tokens)
	return processed_text,span_text


def process_text(s, unicode = False, ptokenizer=tokenizer):
	'''
	Process a text for modeling. Return both a processed version of the text and a description
	of how it was tokenized.
	:param s:
	:return:
	'''

	if unicode:
		processed = s.lower().strip()
	else:
		processed = s.decode('utf-8').lower().strip()
	spans = list(ptokenizer.span_tokenize(processed))
	tokens = ptokenizer.tokenize(processed)
	return tokens,spans

def analyze_column(srs):
	counts,bins=np.histogram(srs,bins=5)
	print 'Distribution of values:'
	print 'Bins: {}'.format(bins)
	print 'Values: {}'.format(counts)
	print 'Fractions: {}'.format(counts/float(srs.shape[0]))


def wiki_revid_to_url(rev_id):
	'''
	Generate a URL for a Wikipedia revision ID
	:param rev_id:
	:return:
	'''
	return 'https://en.wikipedia.org/w/index.php?oldid={}'.format(rev_id)


def reddit_id_to_longint(id):
	'''
	Turns out reddit IDs are base-36, so conversion is easy
	:param id:
	:return:
	'''
	return long(id,36)

def reddit_id_to_url(id,subreddit):
	return 'https://www.reddit.com/r/{}/comments/{}'.format(subreddit, id)

def reddit_utc_to_datetime(utc):
	return datetime.fromtimestamp(utc)


def strip_html(s):
	d = BeautifulSoup(s,'html')
	return d.text


def guardian_time_to_datetime(s):
	'''
	E.g. 25 Nov 2013 16:29
	:param s:
	:return:
	'''
	try:
		return datetime.strptime(s,'%d %b %Y %H:%M')
	except:
		return None


def guardian_url(row):
	'''
	Create a permalink to a guardian comment from a series consisting of the article URL and the comment ID
	:param srs:
	:return:
	'''
	return row['article']+'#comment-'+str(row.name)


def generate_iac_id(row):
	'''
	Generate a numeric ID for a 4forums IAC post. This is dumb, but the dataset doesn't give a
	better unique numeric ID for these comments
	:param row:
	:return:
	'''
	try:
		b1 = bin(int(row['page_id']))[2:]
		b1 = '0'*(16-len(b1)) + b1

		b2 = bin(int(row['tab_number']))[2:]
		b2 = '0'*(16-len(b2)) + b2


		return long(b1+b2,2)
	except:
		try:
			b1 = bin(int(row['page_id.1']))[2:]
			b1 = '0' * (16 - len(b1)) + b1

			b2 = bin(int(row['tab_number.1']))[2:]
			b2 = '0' * (16 - len(b2)) + b2

			return long(b1 + b2, 2)
		except:
			return None


def int_or_none(x):
	try:
		return int(x)
	except:
		return None

def split_params(params, abbreviate_names = True, delimiter='_', only_multiobject_lsts = False, exclude_params=None):
	'''
	Splits a dictionary of parameters where some values are lists, into one set of parameters per combination of listed elements. Gives each one a name.
	:param params: dictionary
	:return: param_sets: a list of tuples. First element of each tuple is a string name for that param set. Second element is a dictionary constituting the set.
	'''

	param_sets = []


	namelsts = [(name,lst) for name,lst in params.items() if type(lst) == list and (not only_multiobject_lsts or len(lst) > 1)]

	if len(namelsts) == 0:
		singleton_params = {k:v[0] if type(v) == list and len(v) == 1 else v for k,v in params.items()}
		return [('all_params', singleton_params, singleton_params)],0

	namelsts = sorted(namelsts, key=lambda x:x[0])

	names, lsts = zip(*namelsts)

	consistent_params = {k:v for k,v in params.items() if k not in names}
	consistent_params.update({k:v[0] for k,v in consistent_params.items() if type(v) == list and len(v) == 1}) #Unpack any singleton list values

	combinations = list(itertools.product(*lsts))

	excluded = 0
	for values in combinations:

		if abbreviate_names:
			combo_name = delimiter.join('{}={}'.format(abbreviate(name), value) for name, value in zip(names, values))
		else:
			combo_name = delimiter.join('{}={}'.format(name,value) for name, value in zip(names, values))

		unique_params = {name:value for name,value in zip(names, values)}
		combo_params = consistent_params.copy()
		combo_params.update(unique_params)

		exclude=False
		if exclude_params:
			for ed in exclude_params:
				if dcontains(combo_params,ed):
					exclude=True

		if not exclude:
			param_sets.append((combo_name, combo_params, unique_params))
		else:
			excluded += 1

		pass

	return param_sets, excluded



def abbreviate(s, split_token = '_'):
	return ''.join(w[0] for w in s.split(split_token))


def dcontains(d1, d2):
	'''
	Check to see if dictionary 2 is contained with dictionary 1
	:param d1:
	:param d2:
	:return:
	'''
	contains = True
	for k,v in d2.items():
		if not (k in d1 and d1[k] == v):
			contains = False
			break
	return contains


ect = pytz.timezone('US/Eastern')
def now():
	now = datetime.now(ect)
	# formatted = now.strftime("%Y-%m-%d %I:%M %p %z")
	# now.strftime = lambda self, format:formatted
	return now

def today():
	now = datetime.now(ect)
	today = now.date()
	return today

iprint_tab_level = 0
iprinting_on = True

def iprint(o='',inc=0,log_info=False):
	if iprinting_on:
		if log_info: print('\t' * (iprint_tab_level+inc) + '\n<{} - {}>'.format(sys.argv[0].split('/')[-1], fdatetime(now())))
		print('\t' * (iprint_tab_level+inc) + str(o).replace('\n', '\n' + '\t' * (iprint_tab_level+inc)))
		if log_info: print('')

def lprint(*args,**kwargs):
	iprint(log_info=True,*args,**kwargs)

def itoggle():
	global iprinting_on
	iprinting_on = not iprinting_on



def iset(n):
	global iprint_tab_level
	iprint_tab_level = n

def iinc():
	global iprint_tab_level
	iprint_tab_level += 1

def idec():
	global iprint_tab_level
	iprint_tab_level -= 1


ticks = []


def tick(comment=None):
	current_time = now()
	if comment:
		iprint('Tick. {} | {}'.format(ftime(current_time), comment))
	else:
		iprint('Tick. {}'.format(ftime(current_time)))
	ticks.append(current_time)


def tock(comment=None):
	'''
	Convenience function for printing a timestamp with a comment
	:param comment:

	:return:
	'''
	last_tick = ticks.pop()
	current_tick = now()

	ps = 'Tock. {}'.format(ftime(current_tick))

	if last_tick is not None:
		ps += ' | {} elapsed.'.format(finterval(current_tick-last_tick))

	if comment:
		ps += ' | {}'.format(comment)

	iprint(ps)


def ftime(dt):
	return dt.strftime("%I:%M %p")

def fdatetime(dt):
	return dt.strftime("%I:%M %p %m/%d/%Y")

def fdatetime_s(dt):
	return dt.strftime("%I:%M:%S %p %m/%d/%Y")

def rdatetime(dtstr):
	return datetime.strptime(dtstr, "%I:%M %p %m/%d/%Y")

def rdatetime2(dtstr):
	return datetime.strptime(dtstr, "%I:%M:%S %p %m/%d/%Y")

def rdatetime3(dtstr):
	return datetime.strptime(dtstr, "%m/%d/%Y %I:%M")

def rdatetime4(dtstr):
	return datetime.strptime(dtstr, "%m/%d/%Y, %I:%M:%S %p")

def rdatetime5(dtstr):
	return datetime.strptime(dtstr, "%d/%m/%Y, %H:%M:%S")

def rdatetime6(dtstr):
	return datetime.strptime(dtstr, "%Y/%m/%d %H:%M:%S")

def rdatetime7(dtstr):
	return datetime.strptime(dtstr, "%d/%m/%Y %H:%M:%S")

def rdatetime8(dtstr):
	return datetime.strptime(dtstr, "%m/%d/%Y %I:%M:%S %p")


def fdatetime_file(dt):
	return dt.strftime("%I.%M_%p_%m-%d-%Y")



def finterval(interval):
	return str(interval)

def remove(item, sequence):
	new_sequence = copy(sequence)
	try:
		new_sequence.remove(item)
	except Exception as x:
		iprint('Warning: could not remove item {} from sequence {}. Error message: {}'.format(item, shortstr(sequence), x.message))
	return new_sequence

def shortstr(o, max_len = 50):
	s = str(o)
	if len(s) > max_len:
		s = s[0:max_len]+'...'
	return s


numbered_filename_pattern = re.compile('([0-9\.]+)_.+')
def highest_current_file_prefix(directory):
	files = os.listdir(directory)
	numbers = []
	for file in files:
		match = re.match(numbered_filename_pattern, file)
		if match:
			numbers.append(float(match.groups()[0]))
	if len(numbers) > 0:
		return max(numbers)
	else:
		return None


def rationale_to_annotation(rationale, tokenization=None):
	annotations = []
	r = 0
	last = None
	current = None
	start = None
	end = None
	while r < len(rationale):
		current = rationale[r]
		if rationale[r] == 1:
			if last == 1:
				end = tokenization[r][1] if tokenization is not None else r+1
			else:
				start, end = tokenization[r] if tokenization is not None else (r,r+1)

		else:
			if last == 1:
				annotations.append((start, end))
			else:
				pass

		last = current
		r += 1

	if last == 1:
		annotations.append((start, end))

	return annotations

def rationales_to_annotation(rationale1, rationale2, r1_name, r2_name, tokenization):
	'''
	Turn two competing rationales into an annotation
	:param rationale1:
	:param rationale2:
	:param r1_name:
	:param r2_name:
	:param tokenization:
	:return:
	'''
	try:
		assert(len(rationale1) == len(rationale2))
	except:
		pass

	combined_rationale = []
	for i in range(len(rationale1)):
		r1i = rationale1[i]
		r2i = rationale2[i]
		if r1i ==1 and r2i == 1:
			combined_rationale.append(1)
		elif r1i == 1 and r2i == 0:
			combined_rationale.append(2)
		elif r1i == 0 and r2i == 1:
			combined_rationale.append(3)
		else:
			combined_rationale.append(0)

	labels = [None, '{}_and_{}'.format(r1_name, r2_name), '{}_only'.format(r1_name), '{}_only'.format(r2_name)]

	annotations = []
	r = 1
	last = combined_rationale[0]
	current = None
	start = tokenization[0][0]
	end = tokenization[0][1]
	while r < len(combined_rationale):
		current = combined_rationale[r]
		if current == last: #if the current value is the same as the last, just extend the current annotation
				end = tokenization[r][1]
		else: #otherwise, we've come to the end of one annotation and the beginning of a new one
			#deal with old one
			if last == 0: #If the values for the old one are 0, skip it
				pass
			else: #otherwise add an annotation with the appropriate label
				annotations.append((labels[last],start, end))

			#deal with new one
			start = tokenization[r][0]
			end = tokenization[r][1]

		last = current
		r += 1

	#Deal with the last annotation
	if last == 0:  # If the values for the old one are 0, skip it
		pass
	else:  # otherwise add an annotation with the appropriate label
		annotations.append((labels[last], start, end))

	return annotations


def test_rationale_to_annotation():
	print 'Testing rationale_to_annotation function()'
	rationale = [1,0,0,1,1,0]

	tokenization = [[0,5],[6,10],[11,15],[16,20],[21,25],[26,30]]

	desired_annotations = [(0,5), (16,25)]

	assert(rationale_to_annotation(rationale, tokenization) == desired_annotations)

	print 'No problems found with rationale_to_annotation function'

# test_rationale_to_annotation()






def annotations_to_rationales(annotation_dict, tokenization):
	'''
	Convert a set of annotations to a set of rationales. Currently using the rule that if part of a token is included, then the whole token is included in the rationale.

	Possible modification would be to see if a majority of the token is included, and only include it in that case. I don't think it will make much difference.

	:param annotation_dict: annotator num --> list of annotation tuples of form (label, relative_span_start, relative_span_end)
	:param tokenization: list of spans defining how the original text was tokenized
	:return: rationale_dict: annotator_num --> single rationale, a list of 0s and 1s the length of the tokenization
	'''

	rationale_dict= {}
	ttree = it.IntervalTree()
	for i, (tstart, tend) in enumerate(tokenization):
		ttree.addi(tstart, tend, data = i)

	for annotator_num, annotation_list in annotation_dict.items():
		rationale_dict[annotator_num] = annotation_to_rationale(annotation_list, tokenization, ttree)

	return rationale_dict


def annotation_to_rationale(annotation_list, tokenization, ttree=None,tokenwise_annotations = False, vector_type=None):
	'''

	:param annotation_list: A list of tuples of the form (label, start, end) or (start, end)
	:param tokenization: A list of character spans which define the tokenization of some text
	:param ttree: An interval tree, if we are running this function on the same text over and over again
	:param tokenwise_annotation: If this is false, then the annotations in annotation_list are character spans. If true, they are token spans.
	:return:
	'''
	rationale = [0 for span in tokenization]

	if not tokenwise_annotations:
		if not ttree:
			ttree = it.IntervalTree()
			for i, (tstart, tend) in enumerate(tokenization):
				ttree.addi(tstart, tend, data=i)

		for annotation_tuple in annotation_list:
			if len(annotation_tuple) == 3:
				label, astart, aend = annotation_tuple
			else:
				astart, aend = annotation_tuple
			for token in ttree.search(astart, aend):
				rationale[token.data] = 1
	else:
		for annotation_tuple in annotation_list:
			astart, aend = annotation_tuple
			for i in range(astart,aend):
				rationale[i] = 1

	if vector_type:
		rationale = np.array(rationale, dtype=vector_type)

	return rationale

def split_list(lst, num):
	'''
	Split a list into num equal-ish sized chunks
	:param lst:
	:param num:
	:return:
	'''

	lsts = [[] for i in range(num)]
	for i in range(len(lst)):
		lsts[(i*num)/len(lst)].append(lst[i])

	lsts = [x for x in lsts if len(x) > 0]

	return lsts


def safe_mean(sequence):
	return np.mean([x for x in sequence if not np.isnan(x)])


def bound(v, minv=0,maxv=1):
	return min(maxv,max(minv,v))

def mean_dict_list(dct_lst, nan_safe = False, prefix=None):
	collected = {}
	for dct in dct_lst:
		for k,v in dct.items():
			if not prefix:
				pk = k
			else:
				pk = prefix+k

			try:
				fv = float(v)
				if not pk in collected:
					collected[pk] = []
				collected[pk].append(fv)
			except:
				pass


	if not nan_safe:
		mean = {k:np.mean(v) for k,v in collected.items()}
	else:
		mean = {k: safe_mean(v) for k, v in collected.items()}



	return mean

class Logger():
	def __init__(self,logfile):
		self.terminal = sys.stdout
		self.log = open(logfile, 'a')
		self.filename = logfile

	def write(self, message):
		self.terminal.write(message)
		# if self.log_dates:
		# 	self.log.write('<{} - {} - {}>\n'.format(sys.argv[0].split('/')[-1], __file__.split('/')[-1], ftime(now())))
		self.log.write(message)
		self.flush()

	def flush(self):
		self.terminal.flush()
		self.log.flush()

	def close(self):
		self.log.close()

	def clear(self):
		self.log.close()
		open(self.filename,'w').close()
		self.log = open(self.filename,'a')


def symlink(src, dest, replace=False):
	if replace and os.path.exists(dest):
		iprint('Replacing existing symlink with new one at {}'.format(dest))
		os.remove(dest)
	os.symlink(src, dest)

def convert_to_distribution(y,output_distribution_interpretation):
	'''
	Convert an nx1 vector of target values to an nxm matrix of target distributions. If the interpreation is "regression", then create one-hot vectors with a 1 in the appropriate bucket.

	If the interpretation is "class_probability", then create 2d vectors with binary class probabilities
	:param y:
	:param output_distribution_size:
	:param output_distribution_interpretation:
	:return:
	'''

	dy = np.empty((len(y), 2))
	for i, yi in enumerate(y):
		try:
			dy[i] = convert_item_to_distribution(yi,output_distribution_interpretation)
		except:
			raise

		# mean_distribution += dy[i]

	mean = 	np.mean(y, axis=0)

	if output_distribution_interpretation == 'class_probability':
		std =  np.sqrt(mean*(1-mean))
	else:
		std = np.std(y,axis=0)

	mean_distribution = np.concatenate([mean, std])


	return dy, mean_distribution

def convert_item_to_distribution(yi,output_distribution_interpretation, minimum_probability=0.01, sigma = 0.1):
	if output_distribution_interpretation == 'class_probability':
		# if output_distribution_size != 2:
		# 	raise Exception('ERROR: Cannot interpret a target value as class probability for any other than two classes')
		#
		# yi = max(minimum_probability, min(1-minimum_probability, float(yi)))
		#
		# dyi = [1-yi, yi]
		dyi = [float(yi), float(max(sigma, np.sqrt(yi*(1-yi))))]
	elif output_distribution_interpretation == 'one_hot':
		# # interv
		# # dyi = [minimum_probability]*output_distribution_size
		# #
		# # dyi[min(int(np.floor(yi*output_distribution_size)),output_distribution_size-1)]=1.0-(output_distribution_size-1)*minimum_probability
		#
		# interval = 1.0 / output_distribution_size
		# bdy = list(frange(interval / 2, 1.0, interval))
		dyi = [yi,sigma]


	return np.matrix(dyi,dtype='float32')

def convert_from_distribution(dy, output_distribution_interpretation):
	y = np.empty((len(dy),1))
	for i,dyi in enumerate(dy):
		y[i] = convert_item_from_distribution(dyi, output_distribution_interpretation)

	return y


def convert_item_from_distribution(dyi, output_distribution_interpretation):
	# if output_distribution_interpretation == 'class_probability':
	# 	if output_distribution_size != 2:
	# 		raise Exception('ERROR: Cannot interpret a target value as class probability for any other than two classes')
	#
	# 	bdy = [0.0,1.0]
	#
	# elif output_distribution_interpretation == 'regression':
	# 	interval = 1.0/output_distribution_size
	# 	bdy = list(frange(interval / 2, 1.0, interval))
	#
	# yi = np.sum(np.asarray(bdy)*dyi)
	# return yi
	return dyi[0]

def frange(start, stop=None, step=1.0, round_to=None):
	if stop == None:
		stop = start
		start = 0.0
	i = start
	while i < stop:
		if round_to is None:
			yield i
		else:
			yield round(i,round_to)
		i += step


class DictClass(OrderedDict):
	'''
	A subclass of OrderedDict that keeps its (ordered) items synced with its attributes
	'''

	def __init__(self, prefix='', *args):

		OrderedDict.__init__(self, args)
		self._prefix = prefix
		self._sync = True

	def __setattr__(self, name, value):
		if hasattr(self, '_sync') and self._sync:
			OrderedDict.__setitem__(self, name, value)

		OrderedDict.__setattr__(self, name, value)

	def __setitem__(self, key, val):
		if hasattr(self, '_sync') and self._sync:
			OrderedDict.__setattr__(self, key, val)
		OrderedDict.__setitem__(self, key, val)

	def __delattr__(self, name):
		if hasattr(self, '_sync') and self._sync:
			del self[name]
		del self.__dict__[name]

	def __str__(self,indent=0):
		rstring = ''
		for key in self.keys():
			if not str(key).startswith('_'):
				rstring += '\t'*indent
				if self._prefix:
					rstring += self._prefix+'_'
				rstring += str(key)+": "
				val = self[key]
				if isinstance(val, DictClass):
					rstring += '\n'+val.__str__(indent = indent+1)
				else:
					rstring += str(val)+'\n'
		return rstring


class SingleItemEvaluation(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.true_y = None
		self.text = None
		self.predicted_rationale_evaluation = RationaleAndPredictionEvaluation(prefix='predicted_rationale')
		self.true_rationale_evaluation = TrueRationaleAndPredictionEvaluation(prefix='true_rationale')
		self.zero_rationale_evaluation = TrivialRationaleAndPredictionEvaluation(prefix='zero_rationale')
		self.one_rationale_evaluation = TrivialRationaleAndPredictionEvaluation(prefix='one_rationale')

	def __str__(self, indent=0, compare_predicted_to_true=True):
		rstring = ''
		for key in self.keys():
			if not str(key).startswith('_'):
				rstring += '\t' * indent
				if self._prefix:
					rstring += self._prefix + '_'
				rstring += str(key) + ": "
				val = self[key]
				if isinstance(val, DictClass):
					if compare_predicted_to_true and key == 'true_rationale_evaluation':
						rstring += '\n' + val.__str__(indent=indent + 1, compareto = self.predicted_rationale_evaluation)
					else:
						rstring += '\n' + val.__str__(indent=indent + 1)
				else:
					rstring += str(val) + '\n'
		return rstring


class DiscreteDatasetEvaluation(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.y_accuracy = None
		self.y_precision = None
		self.y_recall = None
		self.y_f1 = None

		self.rationale_accuracy = None
		self.rationale_precision = None
		self.rationale_recall = None
		self.rationale_f1 = None


class DatasetEvaluation(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.mean_y = None
		self.item_evaluations = []
		self.batch_evaluations = []
		self.mean_predicted_rationale_evaluation = RationaleAndPredictionEvaluation(prefix='mean_predicted_rationale', mean=True)
		self.combined_predicted_rationale_evaluation = DiscreteDatasetEvaluation(prefix='combined_predicted')
		self.mean_true_rationale_evaluation = RationaleAndPredictionEvaluation(prefix='mean_true_rationale', mean=True)
		self.combined_baseline_evaluation = DiscreteDatasetEvaluation(prefix='combined_baseline')
		self.model_properties = ModelProperties()



class RationaleAndPredictionEvaluation(DictClass):
	def __init__(self, prefix='', mean=False):
		DictClass.__init__(self, prefix=prefix)

		self.predicted_y = None
		self.prediction_loss = None
		self.generator_loss = None
		self.encoder_loss = None
		self.inverse_encoder_loss = None
		self.inverse_predicted_y = None
		self.inverse_prediction_loss = None
		self.generator_inverse_loss = None
		self.generator_weighted_inverse_loss = None

		self.rationale = None
		self.probs = None
		self.rationalized_text = None
		self.accuracy = None
		self.precision = None
		self.recall = None
		self.f1 = None
		self.occlusion = None
		self.sparsity_loss = None
		self.weighted_sparsity_loss = None
		self.coherence_loss = None
		self.weighted_coherence_loss = None

		if mean:
			remove = [
				'rationale',
				'rationalized_text',
			]

			for field in remove:
				self.__delattr__(field)

	def __str__(self, indent=0, compareto=None):

		rstring = ''
		for key in self.keys():
			if not str(key).startswith('_'):
				rstring += '\t' * indent
				if self._prefix:
					rstring += self._prefix + '_'
				rstring += str(key) + ": "
				val = self[key]
				if isinstance(val, DictClass):
					rstring += '\n' + val.__str__(indent=indent + 1)
				else:
					if compareto and key in compareto:
						rstring += str(val) + ' ({} {})'.format(compareto[key], compareto._prefix) + '\n'
					else:
						rstring += str(val) + '\n'
		return rstring

class TrueRationaleAndPredictionEvaluation(RationaleAndPredictionEvaluation):
	def __init__(self, prefix='',mean=False):
		RationaleAndPredictionEvaluation.__init__(self, prefix=prefix, mean=mean)

		# Get rid of a few fields that don't really make sense for the true rationales
		remove = [
			'probs',
			'accuracy',
			'precision',
			'recall',
			'f1'
		]

		for field in remove:
			self.__delattr__(field)


class TrivialRationaleAndPredictionEvaluation(RationaleAndPredictionEvaluation):
	def __init__(self, prefix='',mean=False):
		RationaleAndPredictionEvaluation.__init__(self, prefix=prefix, mean=mean)

		keep = [
			'predicted_y',
			'generator_loss',
			'encoder_loss',
			'inverse_encoder_loss',
			'inverse_rationale_predicted_y'
		]

		for key in self.keys():
			if key not in keep:
				self.__delattr__(key)


class ModelProperties(DictClass):
	def __init__(self, prefix=''):
		DictClass.__init__(self, prefix=prefix)

		self.encoder_default_y = None
		self.inverse_encoder_default_y = None
		self.generator_l2_loss = None
		self.generator_l2_weight = None
		self.generator_weighted_l2_loss = None
		self.encoder_l2_loss = None
		self.encoder_l2_weight = None
		self.encoder_weighted_l2_loss = None
		self.inverse_encoder_l2_loss = None
		self.inverse_encoder_l2_weight = None
		self.inverse_encoder_weighted_l2_loss = None
		self.rationale_type = None
		self.prediction_loss_type = None
		self.inverse_encoder_confusion_method = None
		self.inverse_encoder_prediction_loss_type = None
		self.rationale_sparsity_loss_type = None
		self.rationale_sparsity_loss_weight = None
		self.rationale_coherence_loss_type = None
		self.rationale_coherence_loss_weight = None
		self.generator_inverse_encoder_loss_weight = None
		self.generator_inverse_encoder_loss_type = None


def unpad(v, padding_id, x= None):
	'''
	Take a numpy vector and remove all padding from it, returning a reduced vector
	:param v:
	:param padding_id:
	:return:
	'''
	try:
		if x is not None:
			assert(len(x) == len(v))
			return v[x != padding_id]
		else:
			return v[v != padding_id]
	except:
		pass

def dsum(d):
	return '\n'.join([str((k, v.shape, np.sum(v), np.mean(v))) for k, v in d.items()])


def invert(val, func, dtype):
	if func == 'sigmoid':
		return -np.log(1 / val - 1, dtype=dtype)
	elif func == 'tanh':
		return np.arctanh(val, dtype=dtype)
	else:
		raise Exception('Cannot invert unknown function {}'.format(func))

def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def calculate_mcnemar(true_y, py1, py2):
	acc1 = (true_y == py1).astype(int)
	acc2 = (true_y == py2).astype(int)

	a = np.sum((acc1 == 1) & (acc2 == 1))
	b = np.sum((acc1 == 1) & (acc2 == 0))
	c = np.sum((acc1 == 0) & (acc2 == 1))
	d = np.sum((acc1 == 0) & (acc2 == 0))

	if math.fabs(b-c) >0:
		exact = False
	else:
		exact=True

	# if not exact:
	statistic = pow((b-c),2)/float(b+c)
	p_val = stats.chi2.pdf(statistic, 1)
	# else:
	# 	statistic = None
	# 	p_val = 0
	# 	n = b+c
	# 	for i in range(b,n+1):
	# 		p_val += nchoosek(n,i)*pow(0.5,i)*pow(0.5,n-i)
	# 	p_val *= 2



	return statistic,p_val, (a,b,c,d)

def nchoosek(n,k):
	f = math.factorial
	return f(n) / f(k) / f(n - k)


# def send_email(recipient, message, sender, subject=''):
# 	'''
# 	Send an email
# 	:param recipient:
# 	:param message:
# 	:param sender:
# 	:param subject:
# 	:return:
# 	'''
# 	try:
# 		s = smtplib.SMTP('localhost')
# 		msg = MIMEText(message, 'plain')
# 		msg['Subject'] = subject
# 		msg['From'] = sender
# 		msg['To'] = recipient
# 		s.sendmail(sender, [recipient], msg.as_string())
# 		s.quit()
# 	except Exception as ex:
# 		iprint('Could not send email.')
# 		traceback.print_exc()

