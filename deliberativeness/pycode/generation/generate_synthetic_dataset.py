from processing.putil import *
import numpy as np
import nltk
import json
import pandas as pd
import os
import gzip
from pprint import pprint

'''
Generate a synthetic dataset to use to test modeling code
'''

platform = {
  "platform_id": 7,
  "display_title": "Simple synthetic",
  "url": None,
  "start_date": None,
  "end_date": None,
  "description": "Very simple synthetic data",
  "size": 2000

}


dimension_1 = {
  "dimension_id": 5,
  "name": "Good/bad OR",
  "description": "Synthetic label where if there is any bad present, the whole item is labeled bad",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_2 = {
  "dimension_id": 6,
  "name": "Good/bad AND",
  "description": "Synthetic label where the label is proportionate with the amount of bad present",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_3 = {
  "dimension_id": 7,
  "name": "Good/bad/neutral",
  "description": "Synthetic label where the label depends on whether the comment contains the word 'good' or 'bad', and comments consist mostly of the word 'neutral'",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_4 = {
  "dimension_id": 8,
  "name": "Doubled good/bad/neutral",
  "description": "Doubled synthetic label where the label depends on whether the comment contains the word 'good' or 'bad', and comments consist mostly of the word 'neutral'",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_5 = {
  "dimension_id": 9,
  "name": "Short-term context sensitive not bad/bad/neutral",
  "description": "Synthetic dataset where comments consist of the word 'neutral' and one instance of the word 'bad' or phrase 'not bad'",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_6 = {
  "dimension_id": 10,
  "name": "Long-term context sensitive not bad/bad/neutral",
  "description": "Synthetic dataset where comments consist of the word 'neutral' and one instance of the word 'bad' with, in the case of positive instance, the word 'not' somewhere prior to the 'bad'",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_7 = {
  "dimension_id": 11,
  "name": "Only-good and only-bad comments",
  "description": "Bad comments consist of only the word 'bad', and good comments of the word 'good'. The old algorithm should just get one word. The new algorithm should select every word. ",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_8 = {
  "dimension_id": 12,
  "name": "Multiple bad words with conjunctive target",
  "description": "Comments consist of 'good', 'bad_1' and/or 'bad_2'. A comment is only bad if it has both bad_1 and bad_2",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

dimension_9 = {
  "dimension_id": 13,
  "name": "Multiple bad words with disjunctive target",
  "description": "Comments consist of 'good', 'bad_1' and/or 'bad_2'. A comment is bad if it has either bad_1 or bad_2",
  "source_url": None,
  "platform_id": 7,
  "irr": 1.0,
  "size": 1000
}

any_bad_chance = 0.2

#           P(good) P(bad) P(end)
bad_cpd =   [[.9,   .09,   .01], # | good
		   	[.25,   .74,   .01]] # | bad

#           P(good) P(bad) P(end)
good_cpd =   [[.99,  0.0,   .01], # | good
		   	  [None, None,  None]] # | bad


simple_vocabulary = ['good', 'bad']
cpds = [good_cpd, bad_cpd]

outdir = '/data/processed/synthetic'

set_prefixes = ['train','dev','test']
set_fractions = [.7,.2,.1]

p_end_neutral = 0.01

embedding_size = 200
# embeddings = {'good':[1]*50 + [0]*150,
# 			  'bad':[0]*50+[1]*50+[0]*100,
# 			  'not':[0]*100+[1]*50+[0]*50,
# 			  'neutral':[0]*150+[1]*50}


vocabulary = [
	'good',
	'bad',
	'not',
	'neutral',
	'bad_1',
	'bad_2'
]

embeddings = {}
interval = embedding_size/len(vocabulary)
for i in range(0,len(vocabulary)):
	embeddings[vocabulary[i]] = [0]*(i*interval)+[1]*(interval)+[0]*(embedding_size-i*interval-interval)


def main():



	print 'Dumping platform information to {}'.format(os.path.join(outdir,'platform.json'))
	if not os.path.isdir(outdir):
		os.makedirs(outdir)

	with open(os.path.join(outdir,'platform.json'),'w') as f:
		json.dump(platform, f)

	embedding_file_path = os.path.join(outdir, 'embeddings.txt.gz')
	print 'Creating synthetic embedding file at {}'.format(embedding_file_path)
	with gzip.GzipFile(embedding_file_path,'w') as ef:
		for word, embedding_list in embeddings.items():
			embedding_line = ' '.join([word]+[str(float(x)) for x in embedding_list])+'\n'
			ef.write(embedding_line)



	datasets = []
	# datasets.append(('or_target',dimension_1,generate_or_target,generate_synthetic_comment))
	# datasets.append(('and_target',dimension_2,generate_and_target, generate_synthetic_comment))
	# datasets.append(('neutral_target', dimension_3,generate_neutral_target,  generate_neutral_containing_comment))
	# datasets.append(('doubled_neutral_target',dimension_4,generate_neutral_target, generate_doubled_neutral_containing_comment))
	# datasets.append(('short_term_context_sensitive_target', dimension_5, generate_context_sensitive_target, generate_short_term_context_sensitive_comment))
	# datasets.append(('long_term_context_sensitive_target',dimension_6, generate_context_sensitive_target,generate_long_term_context_sensitive_comment))
	# datasets.append(( 'simple_target',  dimension_7,  generate_neutral_target,generate_simple_synthetic_comment))
	datasets.append(('multibad_or', dimension_8, generate_multibad_and_target, generate_multibad_and_comment))
	datasets.append(('multibad_and', dimension_9, generate_multibad_or_target, generate_multibad_or_comment))



	for dir, dimension, target_func, generation_func in datasets:
		print 'Generating {} synthetic dataset'.format(dimension['name'])

		comment_dicts = []
		good_ct = 0
		bad_ct = 0
		for i in range(platform['size']):
			comment = {'datetime':None,
					   'platform_id':platform['platform_id'],
					   'platform_comment_id':i,
					   'url':None}

			mode = int(np.random.rand() <= any_bad_chance)
			if mode == 0:
				good_ct += 1
			else:
				bad_ct += 1

			comment['text'], comment['rationale'] = generation_func(mode)
			tokenizer = nltk.tokenize.WordPunctTokenizer()
			comment['tokenization'] = json.dumps(list(tokenizer.span_tokenize(comment['text'])))
			comment_dicts.append(comment)

		print "{} good comments generated, {} potentially bad. {} total.".format(good_ct, bad_ct, good_ct+bad_ct)


		df = pd.DataFrame(comment_dicts)

		df['original_text'] = df['text']


		path = os.path.join(outdir, dir)
		if not os.path.isdir(path):
			os.makedirs(path)

		print 'Dumping {} version of dataset and dimension info to {}'.format(dir,path)
		df['target'] = df[['text','rationale']].apply(target_func,axis=1)

		with open(os.path.join(path, 'dimension.json'),'w') as f:
			json.dump(dimension, f)


		for i, set, fraction in zip(range(len(set_prefixes)),set_prefixes, set_fractions):
			filepath = os.path.join(path, dir+'_'+set+'.csv')
			start = int(round(df.shape[0]* np.sum(set_fractions[0:i])))
			end = int(round(df.shape[0]*np.sum(set_fractions[0:i+1])))
			print 'Writing {} rows of {} set of {} to {}'.format(end-start,set,dir,filepath)
			df.iloc[start:end].to_csv(filepath)




	print 'Done'


def generate_neutral_target(tr_series):
	return float('bad' in tr_series['text'])

def generate_context_sensitive_target(tr_series):
	if 'not' in tr_series['text'] and 'bad' in tr_series['text']:
		return 0.0
	elif 'not' in tr_series['text'] and 'good' in tr_series['text']:
		return 1.0
	else:
		return generate_neutral_target(tr_series)

def generate_or_target(tr_series):
	return float(np.any(tr_series['rationale']))

def generate_and_target(tr_series, max_frac = 0.25):
	return min(1.0,(np.sum(tr_series['rationale'])/float(len(tr_series['rationale'])))/max_frac)

def generate_multibad_or_target(tr_series):
	if 'bad_1' in  tr_series['text'] or 'bad_2' in tr_series['text']:
		return 1.0
	else:
		return 0.0

def generate_multibad_and_target(tr_series):
	if 'bad_1' in  tr_series['text'] and 'bad_2' in tr_series['text']:
		return 1.0
	else:
		return 0.0


def generate_multibad_comment(p_bad_1=0.5, p_bad_2 = 0.5):
	'''
	Generate a comment consisting mostly of the word 'neutral' with an independent chance to have either or both of 'bad_1' and 'bad_2' somewhere inside.
	:param p_bad_1:
	:param p_bad_2:
	:return:
	'''
	number_of_words = 1
	while np.random.rand() > p_end_neutral:
		number_of_words += 1
	words = ['neutral']*number_of_words
	for bad_word, bad_prob in zip(['bad_1','bad_2'],[p_bad_1, p_bad_2] ):
		if np.random.rand() <= bad_prob:
			insertion_point = int(np.round(np.random.rand() * len(words)))
			words.insert(insertion_point, bad_word)

	rationale = [0 if x == 'neutral' else 1 for x in words]
	return ' '.join(words), rationale

def generate_multibad_and_comment(mode):
	p = np.sqrt(any_bad_chance) #So that p(bad_1 and bad_2) will equal any_bad_chance
	return generate_multibad_comment(p,p)

def generate_multibad_or_comment(mode):
	p = 1-np.sqrt(1-any_bad_chance) # so that #So that p(bad_1 or bad_2) will equal any_bad_chance
	return generate_multibad_comment(p,p)


def generate_synthetic_comment(mode):

	cpd = cpds[mode]
	current_word = sample_cpd(0, cpd)
	while current_word == 2: #if the comment tries to end before having a single word, resample
		current_word = sample_cpd(0, cpd)
	words = []
	rationale = []
	while current_word != 2:
		words.append(simple_vocabulary[current_word])
		rationale.append(current_word)
		current_word = sample_cpd(current_word, cpd)


	return ' '.join(words), rationale


def generate_simple_synthetic_comment(mode):

	word = simple_vocabulary[mode]
	number_of_words = 1
	while np.random.rand() > p_end_neutral:
		number_of_words += 1
	words = [word]*number_of_words
	rationale = [1]*number_of_words


	return ' '.join(words), rationale

def generate_neutral_containing_comment(mode, number_of_insertions = 1, negation_prob = 0, long_term_negation = False):
	relevant_word = simple_vocabulary[mode]
	number_of_neutral_words = 0
	while np.random.rand() > p_end_neutral:
		number_of_neutral_words += 1

	words = ['neutral'] * number_of_neutral_words

	for i in range(number_of_insertions):
		insertion_word = relevant_word

		insertion_point = int(np.round(np.random.rand()*number_of_neutral_words))


		if negation_prob:
			if np.random.rand() < negation_prob:
				if long_term_negation:
					not_insertion_point = int(np.round(np.random.rand()*insertion_point))
					insertion_word = simple_vocabulary[(mode + 1) % 2]
					words.insert(not_insertion_point, 'not')
				else:
					insertion_word = 'not ' + simple_vocabulary[(mode + 1) % 2]

		words.insert(insertion_point, insertion_word)

	rationale = [0 if x == 'neutral' else 1 for x in words]
	return ' '.join(words), rationale


def generate_short_term_context_sensitive_comment(mode):
	return generate_neutral_containing_comment(mode, negation_prob=0.5, long_term_negation=False)

def generate_long_term_context_sensitive_comment(mode):
	return generate_neutral_containing_comment(mode, negation_prob=0.5, long_term_negation=True)

def generate_doubled_neutral_containing_comment(mode):
	return generate_neutral_containing_comment(mode, number_of_insertions=2)

outcomes = [0,1,2]
def sample_cpd(previous, cpd):
	return np.random.choice(outcomes,size=1,p=cpd[previous])[0]


if __name__ == '__main__':
	main()