import os

import pandas as pd

pd.set_option('display.width', 3000)
pd.set_option('max_colwidth',2000)
pd.set_option('precision', 3)


import word2vec
import gzip
import misc.s_options as options
import json
from collections import OrderedDict
# import theano
# import theano.tensor as T


#To suppress annoying scikit-learn warnings

import warnings
warnings.filterwarnings("ignore")

import sklearn.metrics as mt
import numpy as np
np.set_printoptions(linewidth=1000, precision=4, suppress=True, threshold = 2000)
from sklearn.neighbors import BallTree
import processing.putil as pu
from processing.putil import iprint, iinc, idec, itoggle, tick, tock, unpad
from pprint import pformat
import sys

'''
This script trains one or more models on one or more datasets and outputs the test results. Parameter and data sets can be chosen by specifying one
of the python files in the training_params directory. See the readme in that directory for how those files are structured. 

'''



# repo_output_dir = '/output/training_output'
global_output_dir = '../../output/training_output'
output_dir = os.path.join(global_output_dir, '.output')
debug_dir = os.path.join(global_output_dir, '.debugging')

load_existing_pretrained_model = False
load_existing_model = True


#Turn on or off various kinds of analysis
global_analyze_rationale_variants = False #Whether to run a leave-one-out experiment where we try removing contiguous chunks from generated rationales and seeing if doing so improves the objective funtion.
#If it does, that means there's something going pretty wrong with optimization
do_result_comparison = True #Whether to compare results between (only the first two) parameter sets
evaluate_individual_rationale_variations = False #Whether or not to see if variations of a predicted rationale result in lower obejctive function scores
num_rationale_variations = 3
display_related=False #Whether to retrieve and display related examples
display_with_padding = False #Whether to pad display examples as they were padded in training/testing
# classification_metric_average = 'binary'

reduce_rationales = False  # Whether to trim generated rationales down


#Some ways to screw with the training data
double_data_items = False #Whether to take each item from the data and append it to itself to create a doubled version.
minimize_synthetic_rationales = False #Whether to make only the first token of any synthetic rationale the actual rationale
binarize_targets = False
num_reduce =  1000 #Number o
classification_threshold = 0.5 #How to turn a continuous target value into a class value
phrase_capture_threshold = 0 #What percentage of a contiguous phrase has to be captured for the phrasewise measure to consider a hit
tuning_metric = 'pz_f1' #Metric we will tune the hyperparameters to maximize




seed = 109309
# seed = 1234
# seed = 1235


# run_unlabeled_evaluations = False


# sample_unlabeled = 10000


#Some variables that control the display of model results


#Some ways of viewing results
# example_sets = [('High-accuracy high-target-value examples', lambda row: row['target'] > 0.8 and row['squared_error'] < 0.04, 3),
# 				('High-accuracy low-target-value examples', lambda row: row['target'] < 0.2 and row['squared_error'] < 0.04, 3 ),
# 				('Low-accuracy high-target value examples', lambda row: row['target'] > 0.8 and row['squared_error'] > 0.1, 3 ),
# 				('Low-accuracy low-target value examples', lambda row: row['target'] < 0.2 and row['squared_error'] > 0.1, 3 )
# 				]

# example_sets = [('High-target-value examples', lambda row: row['target'] >= 0.7, 15),
# 				# ('Medium-target-value examples', lambda row: row['target'] > 0.3 and row['target'] < 0.7, 3 ),
# 				('Low-target value examples', lambda row: row['target'] <= 0.3 ,3),
# 				]

example_sets = [
	('High target value examples', lambda row: row['target'] >= 0.7, 10),
				# ('Medium target value examples', lambda row: row['target'] > 0.3 and row['target']< 0.7, 10),
				('Low target value examples', lambda row: row['target'] <= 0.3 ,10),
				# ('Low rationale recall examples', lambda row: pd.notnull(row['rationale_recall']) and row['rationale_recall'] <= 0.5 , 10),
				# ('Low rationale precision examples', lambda row: pd.notnull(row['rationale_precision']) and row['rationale_precision'] <= 0.5 , 10),
				# ('Low rationale recall examples', lambda row: pd.notnull(row['rationale_recall']) and row['rationale_recall'] <= 0.5, 10),
				]





print('Identifying a free GPU to use')
from choose_gpu import choose_gpu
gpu_num = choose_gpu(True)


#To change settings based on whether you are doing a remote run from PyCharm or running the script locally fromthe command line
if 'JETBRAINS_REMOTE_RUN' in os.environ:
	print('Running script remotely from PyCharm. Using debugging settings')
	# os.environ["THEANO_FLAGS"]="floatX=float32,optimizer=fast_compile,on_unused_input='warn'"
	# os.environ["THEANO_FLAGS"]="floatX=float32,optimizer=fast_run,on_unused_input='warn'"
	# os.environ["THEANO_FLAGS"]="floatX=float32,optimizer=None,on_unused_input='warn',device=gpu{}".format(gpu_num)
	os.environ["THEANO_FLAGS"]="floatX=float32,optimizer=fast_run,on_unused_input='warn',device=gpu{}".format(gpu_num)
	reduce_data = False
	num_related = 0
	# labeled_sets = labeled_sets[0:2]
	# unlabeled_sets = unlabeled_sets[0:2]
	debugging = True

else:
	print('Running script from command line on Zen. Using Theano normal execution settings')
	# os.environ["THEANO_FLAGS"]="floatX=float32,optimizer=fast_run,on_unused_input='warn'"
	os.environ["THEANO_FLAGS"] = "floatX=float32,optimizer=fast_run,on_unused_input='warn',device=gpu{}".format(gpu_num)
	reduce_data = False
	num_related = 0
	debugging = False


#These imports are down here because we need to set Theano settings before importing anything that uses Theano
from modeling import carton_rationale as r
from modeling.carton_rationale import Model
from original_rationale_dependent import OriginalModel
from sklearn_rationale_model import L2RegressionModel
from lime_rationale_model import LIMEModel
from rationale import myio


print('Theano flags: {}'.format(os.environ["THEANO_FLAGS"]))

def main():
	'''
	Command line file usage: python train_and_evaluate_models.py [script_name] [param_file]
	script_name: short string that describes this run
	param_file: name of param file to use for this run (see training_params directory)
	:return:
	outputs various files to output directory hardcoded above
	'''


	#Hard-code a default parameter file
	from training_params.best_emnlp_model import labeled_sets, unlabeled_sets, all_params, exclude_params

	#Read script name and optional param file from command line
	if len(sys.argv) > 1:
		script_name = sys.argv[1]
		script_dir = os.path.join(output_dir, script_name)

		#Load in experiment parameters based on second command-line argument
		if len(sys.argv > 2):
			if sys.argv[2].startswith('best_emnlp_model'):
				from training_params.best_emnlp_model import labeled_sets, unlabeled_sets, all_params, exclude_params
			elif sys.argv[2].startswith('all_emnlp_lei_variants'):
				from training_params.best_emnlp_lei_variants import labeled_sets, unlabeled_sets, all_params, exclude_params
			elif sys.argv[2].startswith('emnlp_sklearn_model'):
				from training_params import labeled_sets, unlabeled_sets, all_params, exclude_params
			elif sys.argv[2].startswith('emnlp_LIME_model'):
				from training_params.emnlp_LIME_model import labeled_sets, unlabeled_sets, all_params, exclude_params
			else:
				raise Exception('Unrecognized experiment param file. Check the training_params directory for allowable file names')

	else:
		if not os.path.isdir(output_dir):
			os.makedirs(output_dir)
		if not os.path.isdir(debug_dir):
			os.makedirs(debug_dir)
		highest_current_file_prefix = max([pu.highest_current_file_prefix(global_output_dir), pu.highest_current_file_prefix(output_dir), pu.highest_current_file_prefix(debug_dir)])
		if highest_current_file_prefix is None: highest_current_file_prefix = 0
		prefix_pieces = str(float(highest_current_file_prefix)).split('.')
		script_name = '{:03d}'.format(int(prefix_pieces[0]))+'.'+'{:03d}'.format(int(prefix_pieces[1])+1)+'_'+'_'.join([x['prefix'] for x in labeled_sets])+'_'+str(pu.today())
		script_dir = os.path.join(debug_dir,script_name)


	if not os.path.isdir(script_dir):
		os.makedirs(script_dir)


	#Initialize my crummy homemade logger
	logfile = os.path.join(script_dir, 'output.txt')
	logger = pu.Logger(logfile)
	sys.stdout = logger
	sys.stderr = logger
	iprint('Script name set as {}; logging output to {}'.format(script_name, logfile))
	iprint('Using GPU {}'.format(gpu_num))



	all_args = options.load_arguments(all_params)
	assert(isinstance(all_args,options.Arguments))

	iprint('Parameters being used:')
	iprint(pformat(all_params),inc=1)

	iprint('Parameter combinations being excluded:')
	iprint(pformat(exclude_params),inc=1)

	#Split all_params into one dictionary for each combination of values that are lists in all_params
	param_sets, excluded = pu.split_params(all_params, exclude_params=exclude_params)
	minimal_param_sets,_ = pu.split_params(all_params, only_multiobject_lsts=True, exclude_params=exclude_params)
	assert(len(param_sets) == len(minimal_param_sets))

	iprint('Running model training and validation script. Detected {} labeled data sets, {} parameter sets to test for each one, and {} unlabeled datasets to label with each top-performing model.'.format(
		len(labeled_sets),
		len(param_sets),
		len(unlabeled_sets)))

	iprint('{} out of {} parameter combinations were excluded.'.format(excluded, excluded+len(param_sets)))


	overall_start_time = pu.now()
	iprint('Overall start time: {}'.format(overall_start_time))
	# model_id = initial_model_id

	round = 1
	overall_start_time = pu.now()

	# Dump some information about this run to the output directory

	# # ensure that directory exists
	# if not os.path.isdir(overall_output_dir):
	# 	os.makedirs(overall_output_dir)

	with open(os.path.join(script_dir, 'all_params.json'), 'w') as f:
		json.dump(all_params,f)
	with open(os.path.join(script_dir, 'labeled_sets.json'), 'w') as f:
		json.dump(labeled_sets, f,cls=CallableEncoder)
	with open(os.path.join(script_dir, 'unlabeled_sets.json'), 'w') as f:
		json.dump(unlabeled_sets, f)



	# try:
		# raise Exception('Testing error handling')

		#For each labeled dataset
	for labeled_set_num, labeled_set in enumerate(labeled_sets):

		iprint('########################## Labeled set: #{} {} | {} elapsed overall ###########################'.format(labeled_set_num, labeled_set['name'], pu.now() - overall_start_time))
		iinc()
		labeled_set_start_time = pu.now()

		directory_name = labeled_set['prefix']
		global_labeled_directory = os.path.join(global_output_dir, directory_name)
		# ensure that directory exists
		if not os.path.isdir(global_labeled_directory):
			os.makedirs(global_labeled_directory)

		labeled_set_training_results = [] #We'll hold learning curve information for the different models in here
		labeled_set_dev_evaluation_results = [] #We'll hold model evaluation information in here
		labeled_set_test_evaluation_results = [] #We'll hold model evaluation information in here



		# if not has_with(output_json['platforms'], labeled_platform, 'platform_id'):
		# 	output_json['platforms'].append(labeled_platform)


		# output_json['dimensions'].append(dimension)

		with open(os.path.join(global_labeled_directory, 'labeled_set.json'),'w') as f:
			json.dump(labeled_set, f,cls=CallableEncoder)




		#Load in training data
		iprint('Loading in labeled data')
		df_train = pd.read_csv(labeled_set['train'])
		df_dev = pd.read_csv(labeled_set['dev'])
		df_test = pd.read_csv(labeled_set['test'])

		#Convert tokenizations to lists
		for df in [df_train, df_dev, df_test]:
			df['tokenization'] = df['tokenization'].apply(lambda x:json.loads(x))

		if 'dev_rationale' in labeled_set:
			iprint('Adding rationales for some or all development set comments to df')
			dev_rationales_present = True
			df_dev = add_rationales_to_df(df_dev, labeled_set['dev_rationale'])
		else:
			dev_rationales_present = False
			if len(param_sets) > 1:
				iprint("WARNING: No rationales found for development dataset; no way to differentiate rationale performance between the {} parameter sets".format(len(param_sets)))

		if 'test_rationale' in labeled_set:
			iprint('Adding rationales for some or all test set comments to df')
			test_rationales_present = True
			df_test = add_rationales_to_df(df_test, labeled_set['test_rationale'])
		else:
			test_rationales_present = False


		if reduce_data:
			iprint('Sampling {} from each set to speed things up. Remove this when you run the full thing.'.format(num_reduce))
			df_train = df_train.sample(num_reduce, random_state=seed)

			if not dev_rationales_present:
				df_dev = df_dev.sample(num_reduce, random_state=seed)
			else:
				df_dev_with_rationales =  df_dev[df_dev['rationale'].notnull()]
				if df_dev_with_rationales.shape[0] < num_reduce:
					df_dev = pd.concat([df_dev_with_rationales, df_dev.sample(num_reduce-df_dev_with_rationales.shape[0], random_state=seed)]).sample(frac=1)
				else:
					df_dev = df_dev[df_dev['rationale'].notnull()].sample(num_reduce, random_state=seed)

			if not test_rationales_present:
				df_test = df_test.sample(num_reduce, random_state=seed)
			else:
				df_test_with_rationales =  df_test[df_test['rationale'].notnull()]
				if df_test_with_rationales.shape[0] < num_reduce:
					df_test = pd.concat([df_test_with_rationales, df_test.sample(num_reduce-df_test_with_rationales.shape[0], random_state=seed)]).sample(frac=1)
				else:
					df_test = df_test[df_test['rationale'].notnull()].sample(num_reduce, random_state=seed)

		iprint('Resetting indices')
		df_train.reset_index(inplace=True)
		df_dev.reset_index(inplace=True)
		df_test.reset_index(inplace=True)

		iprint('{} comments loaded. {} being used for training, {} for development, {} for testing.'.format(
			df_train.shape[0] + df_dev.shape[0] + df_test.shape[0], df_train.shape[0], df_dev.shape[0],
			df_test.shape[0]))

		#Word vector dimensionality should not be varied in parameters; just do this once per labeled set
		text_srs = pd.concat([df_train['text'], df_dev['text']], axis=0)
		if 'embeddings' in labeled_set:
			embedding_layer = create_or_load_embedding_layer(global_labeled_directory, all_args.embedding_size, text_srs, threads=8, existing_embedding_filepath=labeled_set['embeddings'])
		else:
			embedding_layer = create_or_load_embedding_layer(global_labeled_directory, all_args.embedding_size, text_srs, threads=8)
		padding_id = embedding_layer.vocab_map["<padding>"]

		if double_data_items:
			iprint('Converting every item in the data into a doubled version of itself')
			df_train['text'] = df_train['text'].apply(lambda x: x+' '+x)
			df_dev['text'] = df_dev['text'].apply(lambda x: x + ' ' + x)
			if 'rationale' in df_dev.columns:
				df_dev['rationale'] = df_dev['rationale'].apply(lambda x: np.concatenate((x,x)) if not np.any(np.isnan(x)) else np.NaN)
			df_test['text'] = df_test['text'].apply(lambda x: x + ' ' + x)
			if 'rationale' in df_test.columns:
				df_test['rationale'] = df_test['rationale'].apply(lambda x: np.concatenate((x,x)) if not np.any(np.isnan(x)) else np.NaN)



		if all_args.max_len and all_args.max_len > 0:
			iprint('Restricting length of inputs in training and development set to a max of {} words'.format(all_args.max_len))
			max_len = all_args.max_len
			df_train['text'] = df_train['text'].apply(lambda x: ' '.join(x.split()[0:max_len]))
			df_train['tokenization'] = df_train['tokenization'].apply(lambda x:x[0:max_len])
			df_dev['text'] = df_dev['text'].apply(lambda x: ' '.join(x.split()[0:max_len]))
			df_dev['tokenization'] = df_dev['tokenization'].apply(lambda x:x[0:max_len])

			if 'rationale' in df_dev.columns:
				df_dev['rationale'] = df_dev['rationale'].apply(lambda x: x[0:max_len] if not np.any(np.isnan(x)) else np.NaN)

		else:
			iprint('Not restricting length of input strings')


		if binarize_targets:
			iprint('Binarizing target values to 0 or 1')
			df_train['target'] = df_train['target'].apply(lambda x:float(x >= 0.5))
			df_dev['target'] = df_dev['target'].apply(lambda x: float(x >= 0.5))
			df_test['target'] = df_test['target'].apply(lambda x: float(x >= 0.5))



		df_train['x'] = df_train['text'].apply(lambda x:embedding_layer.map_to_ids(x.split()))
		df_train['y'] = df_train['target']
		train_x = df_train['x'].values
		train_y = df_train['y'].values

		df_dev['x'] = df_dev['text'].apply(lambda x:embedding_layer.map_to_ids(x.split()))
		df_dev['y'] = df_dev['target']
		dev_x = df_dev['x'].values
		dev_y = df_dev['y'].values

		df_test['x'] = df_test['text'].apply(lambda x:embedding_layer.map_to_ids(x.split()))
		df_test['y'] = df_test['target']
		test_x = df_test['x'].values
		test_y = df_test['y'].values



		#We'll track the best rationale model
		best_model = None
		best_dev_evaluation = None
		# best_annotated_dev_df = None
		best_test_evaluation = None
		# best_annotated_test_df = None
		# best_params = None
		best_paramset_name = None


		#This will contain labeled results for each item in the development set, for each parameter set
		annotated_test_dfs = [] #A dataframe with a row for each item in the test set, with a predicted y and rationale and some accuracy metrics
		test_interval_analysis_dfs = [] #Some accuracy metrics over different slices of the test set, sliced by predicted y value
		training_results = [] #A dataframe with a row for each epoch, for the training and dev set, for each paramset, to be used to look at the training curve

		hyperparam_search_start_time = pu.now()
		iprint('Beginning search over {} hyperparameter sets for labeled set {}.'.format(len(param_sets), labeled_set['name']))
		# pretrain_model, pretraining_results = None, None

		#For each unique set of parameters specified in the parameter file
		for paramset_num, ((paramset_name, params, unique_params),(minimal_paramset_name, minimal_params, minimal_unique_params)) in enumerate(zip(param_sets,minimal_param_sets)):

			assert(params == minimal_params)

			iprint('Paramset {} out of {}'.format(paramset_num+1, len(param_sets)))
			iprint('Full paramset name: {}'.format(paramset_name))
			iprint('Short paramset name: {}'.format(minimal_paramset_name))


			iprint('Globally unique params for this run:')
			iprint(pformat(unique_params),inc=1)

			iprint('Locally unique params for this run:')
			iprint(pformat(minimal_unique_params),inc=1)



			iinc()

			# model_id += 1

			global_paramset_directory = os.path.join(global_labeled_directory, paramset_name)
			iprint('Saving information for this parameter set in {}'.format(global_paramset_directory))

			# ensure that directory exists
			if not os.path.isdir(global_paramset_directory):
				os.makedirs(global_paramset_directory)

			paramset_scripts_directory = os.path.join(global_paramset_directory, 'scripts')
			if not os.path.isdir(paramset_scripts_directory):
				os.makedirs(paramset_scripts_directory)

			# put a soft link to this paramset directory in the script output directory and vice versa

			# pu.symlink(script_dir, os.path.join(paramset_scripts_directory, script_name), replace=True)


			# pu.symlink(global_paramset_directory, os.path.join(script_dir, labeled_set['prefix']+'_'+paramset_name), replace=True)


			with open(os.path.join(global_paramset_directory, 'params.json'), 'w') as f:
				json.dump(params, f)

			args = options.load_arguments(params)
			assert(isinstance(args,options.Arguments))

			args.script_name = script_name
			args.paramset_name = paramset_name
			args.short_paramset_name = minimal_paramset_name
			args.save_model = os.path.join(global_paramset_directory,'model.pkl.gz')
			args.load_model = args.save_model

			if args.output_distribution:
				iprint('The encoder will output a distribution over possible discrete classes')

				if 'output_distribution_interpretation' in labeled_set:
					args.output_distribution_interpretation = labeled_set['output_distribution_interpretation']

				if len(train_y[0]) > 1:
					raise Exception("ERROR: Can't have multidimensional target y if we're trying to output distributions.")

			#Subsample training data if necessary
			if args.subsample_training > 0:
				if args.subsample_training <= 1.0:
					sample_size = len(train_x) * args.subsample_training
					iprint('Subsampling training set to {}% of original size {}: {}'.format(100*args.subsample_training, len(train_x), sample_size))
				else:
					sample_size = args.subsample_training
					iprint('Subsampling training set to {} items from original size of {}'.format(sample_size, len(train_x)))
				sample_size = int(min(sample_size, len(train_x)))
				training_indices = list(range(0,len(train_x)))
				np.random.shuffle(training_indices)
				training_indices = training_indices[0:sample_size]
			else:
				iprint('Using entire training set of {} items for training'.format(len(train_x)))
				training_indices = list(range(0,len(train_x)))


			#train model

			if args.model_mode == 'original':
				iprint('Using original model code')
				ModelClass = OriginalModel
			elif args.model_mode == 'new':
				iprint('Using modified model code')
				ModelClass = Model
			elif args.model_mode == 'sklearn':
				iprint('Using simple scikit-learn logistic regression model')
				ModelClass = L2RegressionModel
			elif args.model_mode == 'lime':
				iprint('Using modified code for regression, but LIME for rationales')
				ModelClass = LIMEModel


			if load_existing_model:
				iprint('Loading existing model from {} if possible'.format(args.load_model))

			if not load_existing_model or not os.path.exists(args.load_model):
				raise Exception('We do not want to make a new model')
				iprint('Creating new model')
				model = ModelClass(
						args=args,
						embedding_layer=embedding_layer,
						nclasses=1 if not args.output_distribution else 2,
					)
				model.ready()

				#if the model is supposed to get some pretraining, load in the pretrained model parameters
				# if args.encoder_pretraining_epochs > 0:
				# 	iprint('Loading encoder params from pretrained model')
				# 	model.load_pretrained_encoder(pretrain_model.encoder.params, pretrain_model.encoder_epochs)

			else:
				iprint('Existing model found, so loading it.'.format(args.load_model))
				model = ModelClass(
					args=args,
					embedding_layer=embedding_layer,
					nclasses=-1,
				)
				model.load_model(args.load_model)
				iprint("Model loaded successfully.")


			# repo_paramset_dir = global_paramset_directory.replace(global_output_dir, repo_output_dir)
			# iprint('Found model for this parameter set. Hooray. Copying it (and other stuff) into repo directory {}'.format(repo_paramset_dir))
			# if not os.path.exists(repo_paramset_dir):
			# 	os.makedirs(repo_paramset_dir)
			# 	for file in os.listdir(global_paramset_directory):
			# 		path = os.path.join(global_paramset_directory, file)
			# 		if os.path.isfile(path):
			# 			iprint('Copying {}'.format(file), 1)
			# 			if os.path.exists(os.path.join(repo_paramset_dir, file)):
			# 				iprint('Already exists',2)
			# 			else:
			# 				copy(path, repo_paramset_dir)
			# 				iprint('Done',2)
			# 		else:
			# 			iprint('Ignoring {}'.format(file),1)
			#
			#
			# 	iprint('Successfully copied all contents for this paramset')
			# else:
			# 	iprint('Repo paramset directory already exists, so not copying anything.')



			idec()
			# continue

			# assert_model_loss_function_correct(model)
			train_start_time = pu.now()
			iprint('Training model')
			iinc()
			paramset_training_results = model.train(
				([train_x[i] for i in training_indices], train_y[training_indices]),
				(dev_x, dev_y),
				None,
				dev_rationales = df_dev['rationale'].values if dev_rationales_present else None
			)
			idec()
			iprint('Finished training at {}. {} elapsed overall, {} during training'.format(pu.now(), pu.now() - overall_start_time, pu.now() - train_start_time))

			training_results.extend(paramset_training_results) #Information about which paramset these results are for are already cooked in, as a result of adding that information to args above



			# if training_results and pretraining_results:
			# 	paramset_result_df = pd.DataFrame(pretraining_results + training_results)
			# 	paramset_result_df.to_csv(os.path.join(global_paramset_directory,'training_results.csv'),index=False)



			# iprint('Compiling sampling function')
			# model.dropout.set_value(0.0)


			# stress_test_model(model, embedding_layer, padding_id, args)

			#Run the model over labeled development set
			iprint('Running model over whole development set')
			dev_evaluation, dev_py, dev_pz, annotated_df_dev = run_model_on_set(df=df_dev, df_train=df_train,
											  padding_id=padding_id, args=args, model=model,
											  embedding_layer=embedding_layer, kd=None, ikd=None, known_df=None, display_samples=False, display_prediction_evaluation=False, perform_interval_analysis=False)
			dev_evaluation['paramset_name'] = paramset_name
			dev_evaluation['short_paramset_name'] = minimal_paramset_name
			dev_evaluation.update(minimal_unique_params)

			labeled_set_dev_evaluation_results.append(dev_evaluation)


			iprint('{}/{}: {}'.format(paramset_num+1, len(param_sets), paramset_name))
			iprint('This model has following performance characteristics on development set:')
			iinc()
			iprint(dev_evaluation.combined_performance_string(prefix='Dev '))
			idec()

			iprint('Running model over whole test set')
			test_evaluation, test_py, test_pz, annotated_df_test, test_interval_analysis_df = run_model_on_set(df=df_test, df_train=df_train,
											  padding_id=padding_id, args=args, model=model,
											  embedding_layer=embedding_layer, kd=None, ikd=None, known_df=None, display_samples=False, display_prediction_evaluation=False, analyze_rationale_variants=True, perform_interval_analysis=True)

			test_evaluation['paramset_name'] = paramset_name
			test_evaluation['short_paramset_name'] = minimal_paramset_name
			annotated_test_dfs.append((minimal_paramset_name, annotated_df_test))
			test_evaluation.update(minimal_unique_params)
			labeled_set_test_evaluation_results.append(test_evaluation)
			test_interval_analysis_dfs.append(test_interval_analysis_dfs)


			iprint('{}/{}: {}'.format(paramset_num+1, len(param_sets), paramset_name))
			iprint('This model has following performance characteristics on test set:')
			iinc()
			iprint(test_evaluation.combined_performance_string(prefix='Test '))
			idec()


			#TODO record which model performs best on each metric

			if best_model is None or dev_evaluation[tuning_metric] > best_dev_evaluation[tuning_metric]:
				best_model = model
				best_dev_evaluation = dev_evaluation
				best_annotated_df_dev = annotated_df_dev
				best_test_evaluation = test_evaluation
				best_annotated_df_test = annotated_df_test
				best_paramset_name = paramset_name
				best_test_py = test_py
				best_test_pz = test_pz

			idec()


		# training_result_df = pd.DataFrame(labeled_set_training_results)
		# training_result_df.to_csv(os.path.join(global_labeled_directory, "{}_training_results.csv".format(script_name)),index=False)

		#This should hold development set results for all hyperparameter combinations that were tried.
		dev_evaluation_result_df = pd.DataFrame(labeled_set_dev_evaluation_results)
		dev_evaluation_result_df.to_csv(os.path.join(global_labeled_directory, "{}_dev_evaluation_results.csv".format(script_name)),columns = [column for column in dev_evaluation_result_df.columns if "classification_report" not in column],index=False)

		#This should hold test set evaluation for the best model as well as individual range results for that model
		test_evaluation_result_df = pd.DataFrame(labeled_set_test_evaluation_results)
		test_evaluation_result_df.to_csv(os.path.join(global_labeled_directory, "{}_test_evaluation_results.csv".format(script_name)),columns = [column for column in test_evaluation_result_df.columns if "classification_report" not in column],index=False)

		model = best_model

		args = model.args
		args.save_model = os.path.join(global_labeled_directory, 'model.pkl.gz')
		args.load_model = args.save_model

		# emb_func = model.embedding_function


		# test_model_loading(model, embedding_layer, args)



		iprint('Finished hyperparameter search for labeled set {} in at {} ({} elapsed overall, {} elapsed during search)'.format(labeled_set['name'], pu.now(),(pu.now() - overall_start_time) , (pu.now() - hyperparam_search_start_time)))
		iprint('Best performing model had the following unique parameters: {}'.format(best_paramset_name))
		iprint('Best performing model had the following performance characteristics on development set:')
		iinc()
		iprint(best_dev_evaluation.combined_performance_string(prefix='Best dev '))
		idec()

		iprint('Best performing model had the following performance characteristics on test set:')
		iinc()
		iprint(best_test_evaluation.combined_performance_string(prefix='Best test '))
		idec()



		iprint('Saving best model as {}'.format(args.save_model))
		model.save_model(args.save_model)

		if not reduce_rationales:
			best_annotated_df_test_filepath = os.path.join(global_paramset_directory,'annotated_test_dataset.csv')
		else:
			best_annotated_df_test_filepath =os.path.join(script_dir, 'reduced_annotated_test_dataset.csv')


		iprint('Saving annotated test data for best model to: {}'.format(best_annotated_df_test_filepath))
		best_annotated_df_test.to_csv(best_annotated_df_test_filepath)

		training_result_filepath = os.path.join(script_dir, 'training_results.csv')
		training_result_df = pd.DataFrame(training_results)
		iprint('Saving results of all training epochs to {}'.format(training_result_filepath))
		training_result_df.to_csv(training_result_filepath, index=False, mode='a')

		iprint('Running best model over whole test set again, but with examples this time')
		test_evaluation, test_py, test_pz, annotated_df_test, test_interval_analysis_df = run_model_on_set(df=best_annotated_df_test, df_train=df_train,
																				padding_id=padding_id, args=args, model=model,
																				embedding_layer=embedding_layer, kd=None, ikd=None, known_df=None,
																				display_samples=True,
																				display_prediction_evaluation=False, analyze_rationale_variants=False, perform_interval_analysis=True)


		#Compare the results of two models to see where they differ


		comparison_sets = [
			# ('First model better pz f1 examples', lambda row: row['pz_f1_diff'] > 0.5, 10, 'pz_f1_diff'),
			# ('Seccond model better pz f1 examples', lambda row: row['pz_f1_diff'] < -0.5, 10, 'pz_f1_diff'),
			('Rationales extemely different', lambda row: row['pz_jaccard_similarity'] < 0.25, 10, 'pz_jaccard_similarity'),
						]
		if do_result_comparison:
			tick('Comparing results of parameter sets')
			if len(annotated_test_dfs) < 2:
				iprint('Only 1 parameter set was used, so no comparison can be done')
			else:
				if len(annotated_test_dfs) > 2:
					iprint('Warning: more than two parameter sets were used, so only comparing results of first two')
				short_paramset_name_1, annotated_df_test_1 = annotated_test_dfs[0]
				annotated_df_test_1.columns = [c+'_'+short_paramset_name_1 if not c in df_test.columns else c for c in annotated_df_test_1.columns ]
				short_paramset_name_2, annotated_df_test_2 = annotated_test_dfs[1]
				annotated_df_test_2.columns=[c+'_'+short_paramset_name_2 if not c in df_test.columns else c for c in annotated_df_test_2.columns]

				comparison_df = pd.concat([annotated_df_test_1, annotated_df_test_2[annotated_df_test_2.columns-df_test.columns]],axis=1)
				comparison_df.sort_index(axis=1, inplace=True)
				comparison_df['pz_jaccard_similarity'] = comparison_df[['pz_' + short_paramset_name_1, 'pz_' + short_paramset_name_2]].apply(
					lambda s: mt.jaccard_similarity_score(s.iloc[0], s.iloc[1]), axis=1)

				if 'rationale' in df_test:
					comparison_df['pz_f1_diff'] = comparison_df['rationale_f1_' + short_paramset_name_1]-comparison_df[ 'rationale_f1_' + short_paramset_name_2]

					#Calculate McNemar's statistic for rationale performance
					iprint('Calculating statistical significance of model performance difference')
					combined_true_rationale = []
					combined_pz1 = []
					combined_pz2 = []
					for k in range(df_test.shape[0]):
						true_rationale = df_test['rationale'].iloc[k]
						pz1 = comparison_df['pz_' + short_paramset_name_1].iloc[k]
						pz2 = comparison_df['pz_' + short_paramset_name_2].iloc[k]

						if type(true_rationale) == np.ndarray:
							combined_true_rationale.extend(true_rationale)
							combined_pz1.extend(pz1)
							combined_pz2.extend(pz2)



					assert(len(combined_true_rationale) == len(combined_pz1))
					assert(len(combined_true_rationale) == len(combined_pz2))

					iprint('{} tokens in combined true rationale'.format(len(combined_true_rationale)))

					mcnemar_statistic,p_value, (a,b,c,d) = pu.calculate_mcnemar(np.asarray(combined_true_rationale), np.asarray(combined_pz1), np.asarray(combined_pz2))
					iprint('McNemar statistic value: {}'.format(mcnemar_statistic))
					# p_value = stats.chi2.pdf(mcnemar_statistic, 1)
					iprint('P-value for this value: {}'.format(p_value))
					iprint('Contingency table: 1pos2pos: {}; 1pos2neg: {}; 1neg2pos: {}; 1neg2neg: {}'.format(a,b,c,d))

					acc1 = mt.accuracy_score(combined_true_rationale, combined_pz1)
					prec1 = mt.precision_score(combined_true_rationale, combined_pz1)
					rec1 = mt.recall_score(combined_true_rationale, combined_pz1)
					f1_1 = mt.f1_score(combined_true_rationale,combined_pz1)
					iprint('pz1: acc={}; prec={}; rec={}; f1={}'.format(acc1,prec1,rec1,f1_1))

					acc2 = mt.accuracy_score(combined_true_rationale, combined_pz2)
					prec2 = mt.precision_score(combined_true_rationale, combined_pz2)
					rec2 = mt.recall_score(combined_true_rationale, combined_pz2)
					f1_2 = mt.f1_score(combined_true_rationale,combined_pz2)
					iprint('pz2: acc={}; prec={}; rec={}; f1={}'.format(acc2,prec2,rec2,f1_2))


					iprint('Done calculating significance')




				for set_name, set_rule, num_examples_to_display,set_column in comparison_sets:
					iprint('********************* ' + set_name)

					try:
						set_df = comparison_df[comparison_df.apply(set_rule, axis=1)]
					except Exception as ex:
						iprint('Could not apply rule to set df: {}. Skipping.'.format(ex.message))
					iinc()

					if set_df.shape[0] == 0:
						iprint('No examples found for this set')

					else:
						iprint('{} examples ({:.2f}%) found in this set'.format(set_df.shape[0], (100*set_df.shape[0]/float(comparison_df.shape[0]))))
						set_df = set_df.sample(frac=1, random_state=seed)
						if 'rationale' in set_df.columns:
							set_df['has_rationale'] = set_df['rationale'].notnull()
							set_df.sort(columns='has_rationale', inplace=True, ascending=False)

						for row_num in range(min(num_examples_to_display, set_df.shape[0])):
							set_row = set_df.iloc[row_num]
							nx = set_df['x'].iloc[row_num]  # for some reason pandas tries to convert x into a series if I don't do this
							nzs = [set_df['pz_' + short_paramset_name_1].iloc[row_num],set_df['pz_' + short_paramset_name_2].iloc[row_num]]
							iprint('#{} comparison of {}'.format(row_num + 1, set_name.lower()))
							iprint('{}: {}'.format(set_column, set_row[set_column]))
							iinc()
							for comparison_num,(comparison_paramset_name, comparison_nz) in enumerate(zip([short_paramset_name_1,short_paramset_name_2],nzs)):
								predict_explain_and_display_item(nx=nx, nxrow=set_row, ny=set_row['target'], comment='{}/{} of #{} item of paramset {}'.format(comparison_num+1, len(nzs), row_num + 1, comparison_paramset_name),
															 num_related=num_related, df=df, col='text', model=model, embedding_layer=embedding_layer, kd=None, known_df=None,
															 args=args, padding_id=padding_id, nz=comparison_nz.reshape(len(comparison_nz),1))
							idec()
			tock('Done with parameter set comparison')


		#print a summary of evaluation results for each hyperparameter set
		dev_evaluation_result_df = pd.DataFrame(labeled_set_dev_evaluation_results)
		dev_evaluation_result_df.set_index('short_paramset_name',inplace=True)
		iprint('\nDevelopment set results:')
		iprint(dev_evaluation_result_df[[c for c in dev_evaluation_result_df.columns if not ('classification_report' in c or c=='paramset_name')]].transpose())
		dev_evaluation_result_df.to_csv(os.path.join(script_dir,labeled_set['prefix']+'_dev_evaluation_results.csv'))

		test_evaluation_result_df = pd.DataFrame(labeled_set_test_evaluation_results)
		test_evaluation_result_df.set_index('short_paramset_name',inplace=True)
		iprint('\nTest set results:')
		iprint(test_evaluation_result_df[[c for c in test_evaluation_result_df.columns if not ('classification_report' in c or c=='paramset_name')]].transpose())
		test_evaluation_result_df.to_csv(os.path.join(script_dir,labeled_set['prefix']+'_test_evaluation_results.csv'))

		labeled_set_end_time = pu.now()
		iprint('Labeled set end time: {} ({} elapsed, {} overall elapsed)'.format(labeled_set_end_time, labeled_set_end_time-labeled_set_start_time, labeled_set_end_time - overall_start_time))
		round += 1
		# iprint('Dumping current Django output data to {}'.format(output_json_data_path))
		# write_output_data(output_json, output_json_data_path)
		idec()
		iprint('########################## Finished Labeled set: #{} {} | {}: {} elapsed overall, {} while working with this labeled set'.format(
			labeled_set_num, labeled_set['name'], pu.now(), pu.now() - overall_start_time, pu.now() - labeled_set_start_time))

	# 	process_failed = False
	# except RunException as ex:
	# 	process_failed=True
	# 	iprint('Script failed with following exception:')
	# 	traceback.print_exc()

	iprint('Done at {} ({} elapsed overall).'.format(pu.now(), pu.now()-overall_start_time))
	logger.close()


def do_interval_analysis(annotated_df, buckets=10.0, interval_column='py'):
	'''
	Take a df with predicted labels and rationales, break it up into intervals by target value, and look at performance metrics by interval
	:param annotated_df:
	:param buckets:
	:return:
	'''
	buckets = 10.0
	analysis_intervals = [[i / buckets, (i + 1) / buckets] for i in range(int(buckets))]
	interval_metrics = []
	iprint('Analyzing {} {} intervals of test data'.format(len(analysis_intervals),interval_column))
	iinc()
	for low, high in analysis_intervals:
		if high == 1:  # So that we include 1.0 on the high end
			interval_sub_df = annotated_df[(annotated_df[interval_column] >= low) & (annotated_df[interval_column] <= high)]
		else:
			interval_sub_df = annotated_df[(annotated_df[interval_column] >= low) & (annotated_df[interval_column] < high)]

		iprint('{} examples found in test data with target value in interval {}'.format(interval_sub_df.shape[0], [low, high]))
		interval_sub_df.reset_index(inplace=True)
		if interval_sub_df.shape[0] > 0:
			a_b_y = []
			a_b_py = []
			a_z = []
			a_pz = []
			z_support = 0
			pz = interval_sub_df['pz'].values
			if 'rationale' in interval_sub_df:
				z = interval_sub_df['rationale'].values
			else:
				z= None
			for i, sub_row in interval_sub_df.iterrows():
				a_b_y.append(float(sub_row['target'] > classification_threshold))
				a_b_py.append(float(sub_row['py'] > classification_threshold))
				if z is not None and not np.any(np.isnan(z[i])):
					a_z.extend(z[i])
					a_pz.extend(pz[i])
					z_support += 1



			interval_metrics.append({
				'y_accuracy': mt.accuracy_score(a_b_y, a_b_py),
				'y_precision': mt.precision_score(a_b_y, a_b_py),
				'y_recall': mt.recall_score(a_b_y, a_b_py),
				'y_f1': mt.f1_score(a_b_y, a_b_py),
				'y_support': interval_sub_df.shape[0],
				'z_accuracy': mt.accuracy_score(a_z, a_pz) if z is not None else None,
				'z_precision': mt.precision_score(a_z, a_pz) if z is not None else None,
				'z_recall': mt.recall_score(a_z, a_pz) if z is not None else None,
				'z_f1': mt.f1_score(a_z, a_pz) if z is not None else None,
				'z_support': z_support,
				'mean_py':interval_sub_df['py'].mean(),
				'mean_inverse_py':interval_sub_df['inverse_py'].mean() if 'inverse_py' in interval_sub_df else None,
				'mean_y': interval_sub_df['y'].mean(),
				'{}_low'.format(interval_column): low,
				'{}_high'.format(interval_column): high
			})

	idec()
	interval_analysis_df = pd.DataFrame(interval_metrics)
	iprint('Accuracy metrics by interval:')
	iinc()
	iprint(interval_analysis_df)
	idec()
	return interval_analysis_df


def do_pretrain_model(all_params, train_x, train_y, dev_x, dev_y, labeled_directory, load_existing_pretrained_model, embedding_layer, df_train, df_dev, df_test, param_sets, script_name, labeled_set_training_results, labeled_platform, dimension, model_id, padding_id):
	if 'encoder_pretraining_epochs' in all_params and all_params['encoder_pretraining_epochs'] > 0 and type(all_params['l2_reg']) != list:
		tick('Pretraining one encoder for all hyperparameter combinations')
		iinc()

		pretrained_model_path = os.path.join(labeled_directory, 'pretrained.pkl.gz')

		pretrain_args = options.load_arguments()
		# pretrain_args = argparse.Namespace()
		for k, v in param_sets[0][1].items():
			setattr(pretrain_args, k, v)

		pretrain_args.script_name = script_name
		pretrain_args.paramset_name = 'pretraining'
		pretrain_args.save_model = pretrained_model_path
		pretrain_args.load_model = pretrain_args.save_model

		pretrain_args.split_encoder = False  # This is important.

		if not load_existing_pretrained_model or not os.path.exists(pretrained_model_path):
			iprint('Creating new pretrained model')
			pretrain_model = Model(
				args=pretrain_args,
				embedding_layer=embedding_layer,
				nclasses=len(train_y[0]) if not pretrain_args.output_distribution else 2,
			)
			pretrain_model.ready()
		else:
			iprint('Loading existing pretrained model from {}'.format(pretrained_model_path))
			pretrain_model = Model(
				args=pretrain_args,
				embedding_layer=embedding_layer,
				nclasses=-1,
			)
			pretrain_model.load_model(pretrained_model_path)
			iprint("Pretrained model loaded successfully.")

		iprint('Pretraining encoder')
		iinc()
		pretraining_results = pretrain_model.pretrain(
			(train_x, train_y),
			(dev_x, dev_y),
			None
		)
		labeled_set_training_results.extend(pretraining_results)
		idec()
		tock('Done pretraining encoder')

		iprint('Evaluating prediction accuracy of pretrained encoder without rationales')
		dev_pretrain_evaluation, dev_pretrain_py, _ = run_model_on_set(x=dev_x, y=dev_y, df=df_dev, py=None, pz=None, train_y=train_y, df_train=df_train, df_dev=df_dev,df_test=df_test,padding_id=padding_id, args=pretrain_args, model=pretrain_model, embedding_layer=embedding_layer, kd=None, ikd=None, known_df=None, known_x=None, known_y=None,  model_id=model_id, dimension=dimension, range_id=0, range_boundary=[0, 1], display_samples=False, display_prediction_evaluation=False, no_rationales=True)

		iprint('Pretrained encoder has following performance characteristics on development set:')
		iinc()
		iprint(dev_pretrain_evaluation.prediction_performance_string())
		idec()

		idec()
	else:
		pretraining_results = []
		iprint('Not doing any pretraining')

	return pretrain_model, pretraining_results

def write_output_data(output, path):
	output_json = {}
	for k, lst in output.items():
		output_json[k] = []
		for item in lst:
			if type(item) == dict:
				output_json[k].append(item)
			else:
				output_json[k].append(item.__dict__)

	with open(path, 'wb') as f:
		json.dump(output_json, f)


def generate_rationale_bigram_centroids(bx,bz,emb_func,padding_id):
	return generate_rationale_centroids(bx, bz, emb_func, padding_id, bigram = True)

def generate_rationale_centroids(bx, bz, emb_func, padding_id, seq_pred_func = None, bigram=False):
	'''
	:param bx: len*batch
	:param bz: len*batch
	:param embedding_layer:
	:return: a numpy matrix that is batch*h_size (transposed)
	'''

	be = emb_func(bx)
	if bigram:
		bh = np.zeros([be.shape[1], 2*be.shape[2]])

	else:
		bh = np.zeros([be.shape[1], be.shape[2]])

	if seq_pred_func:
		bspy = seq_pred_func(bx,bz)

	#For each example x
	for j, x in enumerate(bx.T):
		z = bz.T[j]
		e = be[:,j,:]

		if bigram:
			h_sum = np.zeros(2*be.shape[2])
		else:
			h_sum = np.zeros(be.shape[2])

		n = 0.0
		#For each word in x
		for i, we in enumerate(e):
			if z[i] == 1.0 and x[i] != padding_id:

				if seq_pred_func:
					spy = bspy[i, j, 0]
					n += spy
					h_sum += we * spy
				elif bigram:
					n+=1

					if i > 0 and z[i-1] == 1.0 and x[i-1] != padding_id:
						h_sum += np.concatenate([e[i-1],we])
					else:
						h_sum += np.concatenate([np.zeros(be.shape[2]),we])


					if i == (e.shape[0]-1) or z[i+1] == 0.0 or x[i+1] == padding_id:
						n += 1
						h_sum += np.concatenate([we,np.zeros(be.shape[2])])

				else:
					n += 1
					h_sum += we


		if n > 0:
			h_sum /= n

		bh[j] = h_sum

	return bh

def predict_explain_and_display_item(nx=None, nxrow=None, ny=None, comment=None, num_related=None, df=None, col=None, model = None, embedding_layer=None,
									 kd=None, known_df=None, args=None, padding_id=None, nz=None):
	'''
	Make a prediction about and fully explain one example.

	If specified, show explanatory nearest neighbor examples too.
	:return:
	'''
	iprint(comment)
	iinc()
	iprint('Item ID {} from the test set'.format(nxrow.name))



	npy, nz = predict_and_display_item(nx = nx,
									   nxrow = nxrow,
									   ny = ny,
									   col = col,
									   model = model,
									   padding_id=padding_id,
									   embedding_layer = embedding_layer,
									   nz = nz)


	# if num_related > 0 and display_related:
	# 	nh, inh = generate_explanatory_representation(np.reshape(nx, (nx.shape[0], 1)), nz, args, emb_func=emb_func, padding_id=padding_id)
	# 	nnds, nnis = kd.query(nh, num_related)
	# 	# innds, innis = kd.query(inh, k)
	#
	# 	iprint('.....................................................................')
	# 	iprint('Nearest neighbors:')
	# 	iprint('.....................................................................')
	# 	iinc()
	# 	for i, nni in enumerate(nnis[0]):
	# 		nnd = nnds[0, i]
	# 		nnx = known_df.x.values[nni]
	# 		nny = known_df.y.values[nni]
	# 		nnxrow = known_df.iloc[nni]
	# 		iprint('\t{}: distance {}'.format(i, nnd))
	# 		iinc()
	#
	# 		predict_and_display_item(nx=nnx,
	# 								 nxrow=nnxrow,
	# 								 ny=nny,
	# 								 col=col,
	# 								 model=model,
	# 								 embedding_layer = embedding_layer,
	# 								 padding_id=padding_id,
	# 								 prefix='\t')
	# 		idec()
	# 		iprint('.....................................................................')
	# 	idec()
	# else:
	# 	iprint('Not displaying any nearest neighbors for this comment.')
	idec()

	# iprint('.\n'*5)

	return npy, nz


def print_all(dict, prefix,order_list=['py','inverse_py','zero_py','no_z_py','mean_zero_py','encoder_loss', 'generator_loss', 'prediction_loss', 'inverse_generator_prediction_loss', 'inverse_encoder_prediction_loss', 'rationale_sparsity_loss', 'rationale_coherence_loss',  'z', 'gnorm_e', 'gnorm_g', 'gini_impurity_loss'], compareto=None):
	'''
	Print a set of key-value pairs out of a dictionary. Print keys in order of order_list if they are in it.
	:param dict:
	:param prefix:
	:param order_list:
	:param compareto: (dictionary, prefix) tuple to compare this dictionary to
	:return:
	'''
	sorted_items = sorted(dict.items(), key = lambda (k, v): order_list.index(k) if k in order_list else len(order_list))
	for k, v in sorted_items:
		if not compareto:
			try:
				iprint('{} {}: {:.4f}'.format(prefix, k, v))
			except:
				iprint('{} {}: {}'.format(prefix, k, v))
		else:
			try:
				iprint('{} {}: {:.4f} ({:.4f} {})'.format(prefix, k, v, compareto[0], compareto[1]))
			except:
				iprint('{} {}: {} ({} {})'.format(prefix, k, v, compareto[0], compareto[1]))

def predict_and_display_item(nx = None,
		nxrow = None,
		ny = None,
		col = None,
		model = None,
		embedding_layer = None,
		padding_id = None,
		display_inverse=True,
		prefix='',
		display=True,
		nz=None ):
	'''
	Make a prediction about an example x. Display the rationale.
	:param nx:
	:param nxrow:
	:param ny:
	:param col:
	:param z_func:
	:param pred_func:
	:param embedding_layer:
	:return:
	'''
	# assert(isinstance(model,Model))
	iinc()
	if not display: itoggle()
	iprint('Item text:\n {}'.format(nxrow[col]))
	o_nx = nx
	nx = np.reshape(nx, (nx.shape[0],1))

	if nz is None:
		z_sample, inverse_z_sample, n_z_probs = model.rationale_function(nx)
	else:
		nz = np.reshape(nz, (nz.shape[0], 1))
		z_sample = nz
		inverse_z_sample = 1-z_sample
		n_z_probs = z_sample

	prediction_z = z_sample
	inverse_prediction_z = inverse_z_sample
	display_z = z_sample
	inverse_display_z = inverse_z_sample

	if model.args.output_distribution:
		ny = pu.convert_item_to_distribution(ny, model.args.output_distribution_interpretation)
	else:
		ny = np.matrix(ny).astype('float32')

	predicted_evaluation_result= model.evaluation_prediction_function(nx,prediction_z,ny)

	if model.args.output_distribution:
		npy = pu.convert_from_distribution(predicted_evaluation_result['py'], model.args.output_distribution_interpretation)
	else:
		npy = predicted_evaluation_result['py'][0][0]
	nmw = mask_with_rationale(o_nx, display_z, embedding_layer, padding_id, n_z_probs)
	inverse_nmw = mask_with_rationale(o_nx, inverse_display_z, embedding_layer, padding_id, n_z_probs)

	iinc()
	iprint('Actual y: {}'.format(ny))

	iprint(predicted_evaluation_result.prediction_metric_string('Predicted rationale'))

	idec()


	# optimal_nz = optimal_synthetic_z_func(o_nx, embedding_layer)
	#
	# jdist = jaccard(nz, optimal_nz)
	# print prefix, 'Jaccard distance between correct rationale and optimal rationale: {}'.format(jdist)


	# nw = embedding_layer.map_to_words(o_nx)
	# nmw = [w if nz[j] == 1 else '_' * len(w) for j, w in enumerate(nw) if w != '<padding>']
	iprint('Predicted rationale: \n{}'.format(' '.join(nmw)))
	# inverse_nmw = [w if inverse_nz[j] == 1 else '_' * len(w) for j, w in enumerate(nw) if w != '<padding>']
	iprint('Inverted predicted rationale: \n{}'.format(' '.join(inverse_nmw)))

	if 'rationale' in nxrow.index and nxrow['rationale'] is not np.nan:
		true_nz = rationale_to_z_vector(nxrow['rationale'])

		if display_with_padding:
			comparison_z = unpad(display_z, padding_id, nx)
		else:
			comparison_z = display_z

		rationale_evaluation = ModelDiscreteEvaluation()
		rationale_evaluation.update(evaluate_rationale(true_nz, comparison_z,'pz_'))

		if display_with_padding:
			display_true_nz = rationale_to_z_vector(np.pad(nxrow['rationale'],(len(o_nx)-len(nxrow['rationale']), 0),"constant", constant_values=0))
			true_masked_text = mask_with_rationale(o_nx, display_true_nz, embedding_layer, padding_id)
		else:
			true_masked_text = mask_with_rationale(o_nx, true_nz, embedding_layer, padding_id)

		iprint('True rationale: \n{}'.format(' '.join(true_masked_text)))
		iinc()

		iprint(rationale_evaluation.rationale_performance_string())
		true_evaluation_result = model.evaluation_prediction_function(nx, true_nz,ny)
		iprint(true_evaluation_result.prediction_metric_string('True rationale', compareto=(predicted_evaluation_result, 'predicted')))


		if 'generator_loss' in predicted_evaluation_result:
			if true_evaluation_result['generator_loss'] < predicted_evaluation_result['generator_loss']:
				iprint('\tTrue rationale loss was {} less than predicted rationale loss (problem with optimization)'.format(predicted_evaluation_result['generator_loss']-true_evaluation_result['generator_loss']))
			else:
				iprint('\tPredicted rationale loss was {} less than true rationale loss (possible problem with objective function)'.format(true_evaluation_result['generator_loss']-predicted_evaluation_result['generator_loss']))

		idec()
	else:
		iprint('No true rationale present for this item, so not doing rationale evaluation')




	if evaluate_individual_rationale_variations and (display_z == 1.0).any():
		tokenization = list(nxrow['tokenization'])
		annotations = pu.rationale_to_annotation(display_z, tokenization)
		annotation_sets = pu.split_list(annotations, num_rationale_variations)

		iprint('Investigating the loss of {} reduced rationales'.format(len(annotation_sets)))
		iinc()
		num_less = 0
		for reduced_num,reduced_annotations in enumerate(annotation_sets):
			# reduced_annotations = annotations[:i]+annotations[i+1:]

			reduced_nz = rationale_to_z_vector(pu.annotation_to_rationale(reduced_annotations, tokenization))
			reduced_masked_text = mask_with_rationale(o_nx, reduced_nz, embedding_layer, padding_id, None)

			reduced_evaluation_result = model.evaluation_prediction_function(nx, reduced_nz, ny)

			# removed_annotation = annotations[i:i+1]
			# removed_nz = rationale_to_z_vector(pu.annotation_to_rationale(removed_annotation, tokenization))
			# removed_masked_text = mask_with_rationale(o_nx, removed_nz, embedding_layer, padding_id, None)

			iprint('Reduced rationale {} out of {}: \n{}'.format(reduced_num+1,len(annotation_sets),' '.join(reduced_masked_text)))
			# iprint('Removed rationale: \n{}'.format(' '.join(removed_masked_text)))

			iinc()
			iprint(reduced_evaluation_result.prediction_metric_string('Reduced rationale', compareto = (predicted_evaluation_result, 'predicted')))

			if reduced_evaluation_result['generator_loss'] < predicted_evaluation_result['generator_loss']:
				iprint('\tReduced rationale loss was {:.4f} less than predicted rationale loss (problem with optimization)'.format(
					predicted_evaluation_result['generator_loss'] - reduced_evaluation_result['generator_loss']))
				num_less += 1
			else:
				iprint('\tReduced rationale loss was {:.4f} greater than predicted rationale loss (no real conclusion possible)'.format(
					reduced_evaluation_result['generator_loss'] - predicted_evaluation_result['generator_loss']))

			idec()

		percentage_less = float(num_less)/len(annotation_sets)
		iprint('In {} out of {} ({:.1f}%) cases, removing a chunk of the rationale lowered the overall loss. If this is high, then the optimization is not working well.'.format(num_less, len(annotation_sets), 100*percentage_less))
		idec()

		pass
	idec()
	if not display: itoggle()
	return npy,z_sample


def mask_with_rationale(x, z, embedding_layer, padding_id, z_probs = None):
	words = embedding_layer.map_to_words(x)
	#
	# if z_probs is None:
	# 	return [word+'(1.00)' if iz == 1 else '_'*len(word)+'(0.00)' for ix, word, iz in zip(x,words,z) if ix != padding_id]
	# else:
	# 	return [(word if iz == 1 else '_'*len(word))+'({:.2f})'.format(float(iz_prob)) for ix, word, iz, iz_prob in zip(x, words, z, z_probs) if ix != padding_id]

	if z_probs is None:
		return [word+'(1.00)' if iz == 1 else '_'*len(word)+'(0.00)' for ix, word, iz in zip(x,words,z)]
	else:
		return [(word if iz == 1 else '_'*len(word))+'({:.2f})'.format(float(iz_prob)) for ix, word, iz, iz_prob in zip(x, words, z, z_probs)]


def z_to_rationale_spans(z, comment, max_len=None):
	if max_len:
		words = comment.split()[:max_len]
	else:
		words = comment.split()
	assert(len(z) == len(words))
	pzi = 0
	l = 0
	spans = []

	for i, zi in enumerate(z):
		wi = words[i]
		wl = len(wi)+1
		if zi  == 1:
			if pzi == 1:
				spans[-1][1]+= wl
			else:
				spans.append([l,l+len(wi)])


		l += wl
		pzi = zi

	return json.dumps(spans)

def unbatch_batches(batches, include=[], indices = None, transpose = []):
	'''
	Takes in a list of dictionaries whose values are batch results (e.g. py, pz, generator_cost_vector)
	Returns a
	:param batches:
	:param transpose:
	:param indices:
	:return:
	'''
	include = {k:0 for k in include}
	combined_dict = {}

	if indices is not None:
		combined_dict['indices'] = []

	for batch_num, batch in enumerate(batches):
		for key, value in batch.items(): #Assume the batch is a dictionary
			if key in include:
				if key not in combined_dict:
					combined_dict[key] = []

				if key not in transpose:
					combined_dict[key].append(value.ravel())
				else:
					combined_dict[key].append(list(value.T))
		if indices is not None:
			combined_dict['indices'].append(indices[batch_num].ravel())

	unfound_keys = [k for k in include.keys() if include[k] == 0]
	if len(unfound_keys) == 0:
		iprint('Warning: following desired values were not found in the batches:\n{}'.format('\n'.join(['\t'+str(x) for x  in unfound_keys])))

	for key, value in combined_dict.items():
		combined_dict[key] = np.concatenate(combined_dict[key])

	iprint('{} values unbatched. Converting them into a dataframe.'.format(len(combined_dict)) )
	rdf = pd.DataFrame.from_dict(combined_dict)

	if indices is not None:
		rdf.sort_values(by = 'indices', inplace=True)
		rdf.reset_index(inplace=True)

	return rdf



def unbatch_batch(batches, transpose=True):
	items = []
	for batch in batches:
		if transpose:
			for item in batch.T:
				items.append(item)
		else:
			for item in batch:
				items.append(item)
	return items


def evaluate_rationale(true_rationale, predicted_rationale, prefix='', baseline=False):
	'''
	Evaluate how successful a predicted rationale was in comparison to a known rationale
	:param true_rationale:
	:param predicted_rationale:
	:param prefix:
	:return:
	'''
	result = {}
	result['{}true_occlusion'.format(prefix)] = np.mean(true_rationale)
	result['{}predicted_occlusion'.format(prefix)] = np.mean(predicted_rationale)

	result['{}accuracy'.format(prefix)] = mt.accuracy_score(true_rationale, predicted_rationale)
	if not baseline:
		result['{}precision'.format(prefix)] = mt.precision_score(true_rationale, predicted_rationale) if np.any(predicted_rationale) else np.NaN
		result['{}recall'.format(prefix)] = mt.recall_score(true_rationale, predicted_rationale) if np.any(true_rationale) else np.NaN

		normalized_precision  =  mt.precision_score(true_rationale, predicted_rationale) if np.any(predicted_rationale) else 1.0
		normalized_recall = mt.recall_score(true_rationale, predicted_rationale) if np.any(true_rationale) else 1.0
		# result['{}f1'.format(prefix)] = mt.f1_score(true_rationale, predicted_rationale, average=classification_metric_average) if np.any(true_rationale) and np.any(predicted_rationale) else np.NaN
		result['{}f1'.format(prefix)] = 2*normalized_precision*normalized_recall/(normalized_precision + normalized_recall)

		try:
			result['{}classification_report'.format(prefix)] = '\n'+mt.classification_report(true_rationale, predicted_rationale)
		except:
			result['{}classification_report'.format(prefix)] = np.NAN

		#Figure out what percent of true phrases we were able to overlap above a certain threshold (phrasewise recall)
		true_rationale_spans = pu.rationale_to_annotation(true_rationale)
		captured_count = 0
		for start, end in true_rationale_spans:
			token_count = 0
			for i in range(start, end):
				if predicted_rationale[i] == 1:
					token_count += 1
			if token_count / float(end - start) > phrase_capture_threshold:
				captured_count += 1


		#Figure out what percent of predicted phrases overlapped with a true phrase above a certain threshold (phrasewise precision)
		predicted_rationale_spans = pu.rationale_to_annotation(predicted_rationale)
		validated_count = 0
		for start, end in predicted_rationale_spans:
			token_count = 0
			for i in range(start, end):
				if true_rationale[i] == 1:
					token_count += 1
			if token_count / float(end-start) > phrase_capture_threshold:
				validated_count += 1

		result['{}phrase_level_recall'.format(prefix)] = captured_count / float(len(true_rationale_spans)) if len(true_rationale_spans) > 0 else np.NaN
		normalized_phrase_level_recall = captured_count / float(len(true_rationale_spans)) if len(true_rationale_spans) > 0 else 1.0
		result['{}phrase_level_precision'.format(prefix)] = validated_count / float(len(predicted_rationale_spans)) if len(predicted_rationale_spans) > 0 else np.NaN
		normalized_phrase_level_precision = validated_count / float(len(predicted_rationale_spans)) if len(predicted_rationale_spans) > 0 else 1.0

		if normalized_phrase_level_recall == 0 or  normalized_phrase_level_precision ==0:
			result['{}phrase_level_f1'.format(prefix)] = 0
		else:
			result['{}phrase_level_f1'.format(prefix)] = 2*normalized_phrase_level_precision*normalized_phrase_level_recall/(normalized_phrase_level_precision + normalized_phrase_level_recall)


	if np.any(true_rationale):
		pass

	return result


def run_model_on_set(df=None, df_train=None, padding_id=None, args=None, model=None, embedding_layer=None, kd=None, ikd=None, known_df=None, display_samples=False, display_prediction_evaluation=False, display_rationale_evaluation=False, no_rationales=False, analyze_rationale_variants=True, perform_interval_analysis=True):
	'''
	Run and evaluate an existing model on some test/evaluation set. If true labels are available, evaluate accuracy. Either way,
	explain a sample of items from the set and create dictionaries representing django objects and add them to the output list.
	:param display_rationale_evaluation:
	'''

	# assert(isinstance(model, Model))

	iprint('Running and evaluating model')
	eval_result = ModelDiscreteEvaluation()

	if no_rationales:
		iprint('Not using rationales')

	df = df.copy() #We'll make some alterations to this dataframe, so make sure there are no side effects

	batch_size = args.batch
	x_batches, y_batches, i_batches = myio.create_batches(df.x.values, df.y.values, batch_size, padding_id, sort=True,return_indices =True)
	padded_x = unbatch_batch(x_batches, transpose=True)

	#If we didn't pass in pre-existing predictions for the test set, run the z-function and prediction function over the test set
	row_num = 0

	comparison_values = ['inverse_py','generator_loss', 'prediction_loss', 'weighted_prediction_loss', 'sparsity_loss', 'weighted_sparsity_loss', 'coherence_loss', 'weighted_coherence_loss', 'inverse_generator_prediction_loss', 'weighted_inverse_generator_prediction_loss','inverse_encoder_prediction_loss']

	if 'py' not in df or 'pz' not in df:
		# test_py = []
		pz = [None]*df.shape[0]

		# bpys = []
		results = []

		tick('Making predictions about all examples in this set')
		iinc()
		for i, (bx, by, bi) in enumerate(zip(x_batches,y_batches, i_batches)):
			if (i + 1) % 5 == 0:
				iprint('Batch {} of {}'.format(i + 1, len(x_batches)))

			if not no_rationales:

				# if by is None:
				# 	bz , _ , _, _, _ = model.rationale_function(bx)
				# 	bpy = model.prediction_function(bx, bz)[0]
				# else:
				if not args.model_mode == 'lime':
					result = model.itemwise_prediction_function(bx,by)
				else:#lime is so slow, only generate rationales for x's for which there is a true rationale
					btz = df['rationale'][bi.ravel()].notnull().values
					result = model.itemwise_prediction_function(bx,by, btz=btz)
				bz = result['pz']
				# bpy = result['py']
				results.append(result)

				#Numpy vectors/arrays can't safely be stored in dataframe cells
				unpadded_bz = [unpad(z, padding_id, bx[:, j]) for j, z in enumerate(list(bz.T))]
				for original_index, z in zip(bi, unpadded_bz):
					pz[int(original_index)] = z

				# for k,z in enumerate(unpadded_bz):
				# 	row = df.iloc[row_num]
				#
				# 	if 'rationale' in df and type(row['rationale']) == np.ndarray:
				# 		if len(z) != len(row['rationale']):
				# 			pass
				# 	row_num += 1


				# pz.extend(unpadded_bz)
			else:
				bpy = model.no_z_prediction_function(bx)[0]

			if (args.output_distribution):
				bpy = pu.convert_from_distribution(bpy, args.output_distribution_interpretation)
			# bpys.append(bpy)


		idec()
		# n_test_x = np.vstack(tbxs)
		# test_pz = np.vstack(tbzs)
		# py = np.vstack(bpys)

		# result_df = unbatch_batches(results, include=['generator_loss'], indices = i_batches)
		result_df = unbatch_batches(results, include=['py']+comparison_values, indices = i_batches)

		assert(not np.any([pzi is None for pzi in pz]))

		df['pz'] = pz
		assert(np.all(df['pz'].apply(len) == df['x'].apply(len)))

		# df['py'] = py
		df = pd.concat([df, result_df],axis=1)
		py = df['py'].values


		tock()
	else:
		iprint('Test predictions and rationales have been provided, so not regenerating them.')
		pz = df['pz'].values
		py = df['py'].values


	if (global_analyze_rationale_variants and analyze_rationale_variants) or reduce_rationales:

		if global_analyze_rationale_variants:
			iprint('Evaluating what percentage of contiguous rationale chunks could be discarded with improvement to the loss function')
		elif reduce_rationales:
			iprint('Reducing every rationale to just the most helpful phrase')

		#assemble all variations of all predicted rationales into one master list
		tick('Generating variants')
		variants = []
		tokenization = df['tokenization'].values
		x=df['x'].values
		iinc()
		for i, row in df.iterrows():
			if (i+1) % (df.shape[0]/10) == 0:
				iprint('{}/{}...'.format(i+1, df.shape[0]))

			annotations = pu.rationale_to_annotation(pz[i])

			if len(annotations) > 1:
				for chunk in annotations:
					if global_analyze_rationale_variants:
						reduced_annotation = annotations[:]
						reduced_annotation.remove(chunk)
					elif reduce_rationales:
						reduced_annotation = [chunk]

					reduced_rationale = pu.annotation_to_rationale(reduced_annotation, tokenization[i],  tokenwise_annotations=True, vector_type = 'float32')


					variants_dict = {
						'original_i':i,
						'x': x[i],
						'y': row['target'],
						'z':pz[i],
						'|z|':np.sum(pz[i]),
						'|z_chunks|': len(annotations),
						'py': row['py'],
						'reduced_z':reduced_rationale,
						'|reduced_z|':np.sum(reduced_rationale),
						'|reduced_z_chunks|':len(reduced_annotation)
					}
					for value in comparison_values:
						if value in row:
							variants_dict[value] = row[value]

					variants.append(variants_dict)

		idec()
		if len(variants) > 0:
			variant_df = pd.DataFrame(variants)
			tock('{} variants generated. Rationales reduced from an average of {} chunks to an average of {} chunks'.format(variant_df.shape[0], variant_df['|z_chunks|'].mean(),variant_df['|reduced_z_chunks|'].mean() ))
			#batch all the variations up and run them through an evaluation function, looking for mainly just the value of generator_loss

			variant_batches_x, variant_batches_y, variant_batches_z, variant_batches_i = myio.create_batches(variant_df['x'].values, variant_df['y'].values, args.batch, padding_id, return_indices=True,z=variant_df['reduced_z'].values)

			iprint('Evaluating {} batches of reduced rationales'.format(len(variant_batches_x)))
			iinc()
			variant_batch_results = []
			for batch_num,(variant_batch_x, variant_batch_y, variant_batch_z, variant_batch_i) in enumerate(zip(variant_batches_x, variant_batches_y, variant_batches_z, variant_batches_i)):
				if (batch_num+1) % 5 == 0:
					iprint('Variant batch {} of {}'.format(batch_num+1, len(variant_batches_x)))
				variant_batch_result = model.itemwise_evaluation_prediction_function(variant_batch_x.astype('int32'), variant_batch_z.astype('float32'), variant_batch_y)
				variant_batch_results.append(variant_batch_result)
			idec()

			variant_result_df = unbatch_batches(variant_batch_results, include=['py']+comparison_values, indices=variant_batches_i)
			variant_result_df.rename(columns = lambda c:'reduced_'+c, inplace=True)
			iprint('Aggregate results of reducing rationales:')
			iinc()
			variant_df = pd.concat([variant_df, variant_result_df],axis=1)
			for value in comparison_values:
				diff = variant_df['reduced_'+value] - variant_df[value]
				mean_diff = diff.mean()
				percent_improved = 100*(diff < 0).mean()
				diff_string = ('' if mean_diff < 0 else '+') +'{:.6f}'.format(mean_diff)
				gbstring = 'IMPROVED' if mean_diff < 0 else 'WORSENED'
				iprint('{}: {} average, improved {:.1f}% of the time ({})'.format(value, diff_string, percent_improved, gbstring))

			idec()

			if reduce_rationales:

				reduced_variant_df = variant_df.sort_values('reduced_generator_loss').groupby('original_i',as_index=False).first()
				iprint('Found a reduced rationale for {} out of {} rationales in the original set. Replacing originals with these.'.format(reduced_variant_df.shape[0], df.shape[0]))
				for i, row in reduced_variant_df.iterrows():
					pz[row['original_i']] = reduced_variant_df['reduced_z'].iloc[i]
				df['pz'] = pz
				#group the df by original_i, and then select the variant for each i with the smallest generator loss
				#then replace the generated rationale in pz with that reduced rationale
				pass

			# variant_df['improved'] = variant_df['reduced_generator_loss'] < variant_df['generator_loss']
			# overall_fragment_redundancy = variant_df['improved'].mean()
			#
			# iprint('{:.3f}% of all rationale fragments can be discarded to lower the overall loss'.format(100*overall_fragment_redundancy))

			pass
		else:
			iprint('Could not find any variants, so not doing any variant analysis.')
		#unbatch all the variations and re-associate them from the rationale from which they were generated

		#Figure out the percentage of unnecessary chunks, and the average percentage of unnecessary chunks by comparing the generator loss of the variations with that of the original rationale for each set of variants



	#If ground truth values are provided for the training and test sets, evaluate the regression and classification performance of the model
	if df_train is not None and 'y' in df:
		mean_y_val = df_train.y.mean()
		mean_py = np.ones(df.y.values.shape) * mean_y_val
		binarized_mean_py = (mean_py > classification_threshold).astype(float)
		binarized_y = (df.y.values > classification_threshold).astype(float)
		binarized_py = (py > classification_threshold).astype(float)

		eval_result['py_accuracy'] = mt.accuracy_score(binarized_y, binarized_py)
		eval_result['py_f1'] = mt.f1_score(binarized_y, binarized_py)
		eval_result['py_precision'] = mt.precision_score(binarized_y,binarized_py)
		eval_result['py_recall'] = mt.recall_score(binarized_y,binarized_py)
		eval_result['py_classification_report'] = '\n'+mt.classification_report(binarized_y,binarized_py)
		eval_result['py_mse'] = mt.mean_squared_error(df.y.values, py)
		eval_result['py_mae'] = mt.mean_absolute_error(df.y.values, py)

		eval_result['mcy_accuracy'] = mt.accuracy_score(binarized_y, binarized_mean_py)
		# eval_result['mcy_f1'] = mt.f1_score(binarized_y, binarized_mean_py)
		# eval_result['mcy_precision'] = mt.precision_score(binarized_y,binarized_mean_py)
		# eval_result['mcy_recall'] = mt.recall_score(binarized_y,binarized_mean_py)
		# eval_result['mcy_classification_report'] = '\n'+mt.classification_report(binarized_y,binarized_mean_py)
		eval_result['mcy_mse'] = mt.mean_squared_error(df.y.values, mean_py)
		eval_result['mcy_mae'] = mt.mean_absolute_error(df.y.values, mean_py)


		if display_prediction_evaluation:
			iprint('Model prediction performance:')
			# test_y = test_y[:, args.aspect]
			# train_y = train_y[:, args.aspect]

			iinc()
			iprint(eval_result.prediction_performance_string())
			idec()

			# if 'rationale' in df.columns:
			# 	iprint('There are rationale labels, so evaluating rationale accuracy:')
			# 	similarities = [jaccard(z, pz) for z, pz in zip(df['rationale'].apply(json.loads), pz)]
			# 	mean_jaccard = np.mean(similarities)
			# 	iprint('Mean jaccard distance between true and predicted rationales: {}'.format(mean_jaccard))
	else:
		iprint('True y values not provided, so not doing evaluation of classifier')


	if display_prediction_evaluation:
		iprint('Distribution of actual values:')
		counts, bins = np.histogram(df.y.values, bins=[0., 0.2, 0.4, 0.6, 0.8, 1.])
		for i, count in enumerate(counts):
			iprint('\t[{}-{}]: {}({:.2f}%)'.format(bins[i], bins[i + 1], count, 100 * float(count) / len(df.y.values())))

		#Even with no ground truth, we can still look at the distribution of predicted values
		iprint('Distribution of predicted values:')
		counts, bins = np.histogram(py, bins=[0., 0.2, 0.4,  0.6,  0.8, 1.])
		for i, count in enumerate(counts):
			iprint('\t[{}-{}]: {}({:.2f}%)'.format(bins[i], bins[i + 1], count, 100 * float(count) / len(py)))

	#If there is a "rationale" column in df_test, then evaluate pz with respect to corresponding rationales in the data
	if 'rationale' in df.columns and not no_rationales:
		iprint('Rationale column is present in df, so evaluating rationales')
		iprint('Using {} as the phrase capture threshold for phrase-wise metrics'.format(phrase_capture_threshold))
		combined_true_rationale = []
		combined_predicted_rationale = []
		rcount = 0
		rationales = df[df['rationale'].notnull()]['rationale']
		mean_z =  np.mean([zi  for rationale in rationales for zi in rationale])
		mcz = int(mean_z >= 0.5)
		iprint('Most common rationale value: {}'.format(mcz))
		rationale_evaluations = []
		mcz_rationale_evaluations = []
		df['rationale_f1'] = None
		for i in range(df.shape[0]):
			true_rationale = df['rationale'].iloc[i]
			if type(true_rationale) == np.ndarray: #that is, if this row has a rationale
				# if args.max_len > 0:
				# 	true_rationale = true_rationale[0:args.max_len]
				if len(true_rationale) != len(pz[i]):
					raise Exception('Predicted rationale for row {} has {} tokens, while true rationale has {} tokens'.format(i, len(pz[i]), len(true_rationale)))

				rcount +=1

				combined_true_rationale.extend(true_rationale)
				combined_predicted_rationale.extend(pz[i])

				rationale_evaluation = evaluate_rationale(true_rationale, pz[i])
				df.loc[i,'rationale_accuracy'] = rationale_evaluation['accuracy']
				df.loc[i,'rationale_f1'] = rationale_evaluation['f1']
				df.loc[i, 'rationale_precision'] = rationale_evaluation['precision']
				df.loc[i, 'rationale_recall'] = rationale_evaluation['recall']


				df.loc[i,'rationale_phrase_level_f1'] = rationale_evaluation['phrase_level_f1']
				df.loc[i,'rationale_phrase_level_recall'] = rationale_evaluation['phrase_level_recall']
				df.loc[i,'rationale_phrase_level_precision'] = rationale_evaluation['phrase_level_precision']



				rationale_evaluations.append(rationale_evaluation)

				#Calculate how successful we'd be by just using the most-common value for the rationale, as a baseline
				mcz_rationale = [mcz]*len(true_rationale)
				mcz_rational_evaluation = evaluate_rationale(true_rationale, mcz_rationale, baseline=True)
				mcz_rationale_evaluations.append(mcz_rational_evaluation)


		mean_rationale_evaluation = pu.mean_dict_list(rationale_evaluations,nan_safe=True, prefix='pz_mean_')
		combined_rationale_evaluation = evaluate_rationale(combined_true_rationale, combined_predicted_rationale, 'pz_')
		eval_result.update(mean_rationale_evaluation)
		eval_result.update(combined_rationale_evaluation)

		mcz_mean_rationale_evaluation = pu.mean_dict_list(mcz_rationale_evaluations,nan_safe=True,prefix='mcz_mean_')
		combined_mcz_rationale = [mcz]*len(combined_true_rationale)
		mcz_combined_rationale_evaluation = evaluate_rationale(combined_true_rationale, combined_mcz_rationale, 'mcz_',baseline=True)
		eval_result.update(mcz_mean_rationale_evaluation)
		eval_result.update(mcz_combined_rationale_evaluation)



		if display_rationale_evaluation:
			iprint('Rationale performance:')
			iinc()
			iprint(eval_result.rationale_performance_string())
			idec()



	#If directed, and if there are ground truth values, then display explanations for a few semi-randomly selected items from the test set
	if display_samples and 'target' in df.columns and not no_rationales:
		iprint('Explaining a few random examples from the test data. Preferring examples with known rationales')

		if display_with_padding:
			df['x'] = padded_x

		df['squared_error'] = (df.y.values-py)**2

		for set_name, set_rule, num_examples_to_display in example_sets:
			iprint('********************* '+set_name)

			set_df = df[df.apply(set_rule, axis=1)]
			iinc()

			if set_df.shape[0] == 0:
				iprint('No examples found for this set')

			else:
				set_df = set_df.sample(frac=1, random_state=seed)
				if 'rationale' in set_df.columns:
					set_df['has_rationale'] = set_df['rationale'].notnull()
					set_df.sort(columns='has_rationale',inplace=True,ascending=False)

				for row_num in range(min(num_examples_to_display, set_df.shape[0])):
					set_row = set_df.iloc[row_num]
					nx = set_df['x'].iloc[row_num] #for some reason pandas tries to convert x into a series if I don't do this

					if reduce_rationales: #if we reduced rationales earlier, use the existing reduced ones. If not, regenerate them so we can see the probabilities as well.
						nz = set_df['pz'].iloc[row_num]
					else:
						nz = None

					predict_explain_and_display_item(nx=nx, nxrow=set_row, ny=set_row['target'], comment='#{} item of {}'.format(row_num+1, set_name.lower()), num_related=num_related, df=df, col='text', model=model, embedding_layer=embedding_layer, kd=kd, known_df=known_df, args=args, padding_id=padding_id, nz = nz)


			idec()




	#Analyze model performance by interval
	if perform_interval_analysis:
		interval_analysis_df = do_interval_analysis(df)




	if perform_interval_analysis:
		return eval_result, py, pz, df, interval_analysis_df
	else:
		return eval_result, py, pz, df





class ModelDiscreteEvaluation(OrderedDict):
	'''
	Class that holds the result of running a model over a set
	'''
	def __init__(self):

		super(ModelDiscreteEvaluation, self).__init__()

		self['py_accuracy'] = np.nan  # Binarized accuracy of py
		self['py_f1'] = np.nan  # Binarized f1 of py
		self['py_precision'] = np.nan  # Binzarized precision of py
		self['py_recall'] = np.nan  # Binarized recall of py
		self['py_mse'] = np.nan  # Mean squared error of py
		self['py_mae'] = np.nan  # Mean absolute error of py

		self['mcy_accuracy'] = np.nan  # Binarized accuracy of mcy
		# self['mcy_f1'] = np.nan  # Binarized f1 of mcy
		# self['mcy_precision'] = np.nan  # Binzarized precision of mcy
		# self['mcy_recall'] = np.nan  # Binarized recall of mcy
		self['mcy_mse'] = np.nan  # Mean squared error of mcy
		self['mcy_mae'] = np.nan  # Mean absolute error of mcy


		self['pz_accuracy'] = np.nan  # Overall tokenwise accuracy of pz
		self['pz_f1'] = np.nan  # Overall tokenwise f1 of pz
		self['pz_precision'] = np.nan  # Overall tokenwise accuracy of pz
		self['pz_recall'] = np.nan  # Overall tokenwise f1 of pz
		self['pz_classification_report'] = None  # Classification report for pz
		self['pz_phrase_level_precision'] = np.nan  # What percentage of contiguous true rationale phrases are at least partially captured
		self['pz_phrase_level_recall'] = np.nan  # What percentage of contiguous true rationale phrases are at least partially captured
		self['pz_phrase_level_f1'] = np.nan

		self['pz_mean_accuracy'] = np.nan  # Mean tokenwise accuracy of pz
		self['pz_mean_f1'] = np.nan  # Mean tokenwise f1 of pz
		self['pz_mean_precision'] = np.nan  # Mean tokenwise accuracy of pz
		self['pz_mean_recall'] = np.nan  # Mean tokenwise f1 of pz
		self['pz_mean_phrase_level_precision'] = np.nan  # Mean phrase-level f1 of pz
		self['pz_mean_phrase_level_recall'] = np.nan  # Mean phrase-level f1 of pz
		self['pz_mean_phrase_level_f1'] = np.nan



		self['mcz_accuracy'] = np.nan  # Overall tokenwise accuracy of mcz
		# self['mcz_f1'] = np.nan  # Overall tokenwise f1 of mcz
		# self['mcz_precision'] = np.nan  # Overall tokenwise accuracy of mcz
		# self['mcz_recall'] = np.nan  # Overall tokenwise f1 of mcz
		# self['mcz_classification_report'] = None  # Classification report for mcz


		# self['mcz_mean_accuracy'] = np.nan  # Mean tokenwise accuracy of mcz
		# self['mcz_mean_f1'] = np.nan  # Mean tokenwise f1 of mcz
		# self['mcz_mean_precision'] = np.nan  # Mean tokenwise accuracy of mcz
		# self['mcz_mean_recall'] = np.nan  # Mean tokenwise f1 of mcz
		# self['mcz_mean_phrase_level_recall'] = np.nan  # Mean phrase-level f1 of mcz


		self.prediction_metrics = [
			'py_accuracy',
			'py_f1',
			'py_mse',
			'py_mae',
			'py_precision',
			'py_recall',
			'mcy_accuracy',
			# 'mcy_f1',
			'mcy_mse',
			# 'mcy_mae',
			# 'mcy_precision',
			# 'mcy_recall'
			]

		self.rationale_metrics = [
			'pz_accuracy',
			'pz_f1',
			'pz_precision',
			'pz_recall',
			'pz_phrase_level_precision',
			'pz_phrase_level_recall',
			'pz_phrase_level_f1',
			'pz_classification_report',
			'pz_mean_accuracy',
			'pz_mean_f1',
			'pz_mean_precision',
			'pz_mean_recall',
			'pz_mean_phrase_level_precision',
			'pz_mean_phrase_level_recall',
			'pz_mean_phrase_level_f1',
			'mcz_accuracy',
			# 'mcz_f1',
			# 'mcz_precision',
			# 'mcz_recall',
			# 'mcz_phrase_level_recall',
			# 'mcz_classification_report',
			# 'mcz_mean_accuracy',
			# 'mcz_mean_f1',
			# 'mcz_mean_precision',
			# 'mcz_mean_recall',
			# 'mcz_mean_phrase_level_recall'
		]

		self.all_metrics = self.prediction_metrics+self.rationale_metrics

	def expand_acronyms(self,s):
		e = s.replace('_', ' ')
		e = e.replace('py','Predicted y')
		e = e.replace('mcy','Most-common/mean y')
		e = e.replace('pz','Predicted rationale')
		e = e.replace('mcz','Most-common rationale')


		return e



	def rationale_performance_string(self,prefix=''):
		return self.__str__(keys=self.rationale_metrics,prefix=prefix)

	def prediction_performance_string(self,prefix=''):
		return self.__str__(keys = self.prediction_metrics,prefix=prefix)

	def combined_performance_string(self,prefix=''):
		return self.__str__(keys = self.all_metrics,prefix=prefix)

	def __str__(self,keys = None,prefix=''):
		if keys == None:
			keys = self.keys()
		strs = []
		for k in keys:
			if k not in self:
				strs.append('{}: <NC>'.format(k))
			else:
				v = self[k]
				if v is not None and not (not np.isscalar(v) and np.any(np.isnan(v))):
					try:
						strs.append("{}{}: {:.3f}".format(prefix,self.expand_acronyms(k), float(v)))
					except:
						strs.append("{}{}: {}".format(prefix,self.expand_acronyms(k), v))

		return '\n'.join([s for s in strs])



def has_with(lst, dct, key):
	'''
	Returns true if list has a dictionary whose value for key is the same value of dct[key]
	:param lst:
	:param dct:
	:param key:
	:return:
	'''
	for item in lst:
		if key in item:
			if item[key] == dct[key]:
				return True
	return False


def jaccard(primary_seq, secondary_seq):
	# if not len(primary_seq) == len(secondary_seq):
		# raise Exception()
	return mt.jaccard_similarity_score(primary_seq, secondary_seq[0:len(primary_seq)])




def generate_explanatory_representations(train_x, dev_x, train_y, dev_y, df_train, df_dev, emb_func, padding_id, model, args):
	# If we are going to be calling on the model to retrieved explanatory cases, cache all training and validation examples into a space partitioning data structure
	if num_related > 0:

		if args.retrieval == 'output_weighted_rationale_centroid':
			iprint('Compiling sequence preduction function')
			seq_pred_func = model.sequence_prediction_function

		iprint('Generating explanatory representations for training and validation data and sticking them in a space partitioning data structure')
		iinc()
		explanatory_rep_start_time = pu.now()

		known_x = train_x + dev_x
		known_y = np.concatenate((train_y, dev_y))
		known_df = pd.concat((df_train, df_dev))
		known_df.reset_index(inplace=True)

		known_x_batches, known_y_batches = myio.create_batches(known_x, known_y, args.batch, padding_id, sort=False)
		bhs = []
		ibhs = []
		bxs = []
		bzs = []
		for i, bx in enumerate(known_x_batches):
			if (i + 1) % 5 == 0:
				iprint('Batch {} of {} @ {}'.format(i + 1, len(known_x_batches), pu.now()))
			bz, _, _ = model.mle_z_func(bx)
			bh, ibh = generate_explanatory_representation(bx, bz, args, emb_func=emb_func, padding_id=padding_id)
			# by = pred_func(bx, bz)
			bhs.append(bh)
			ibhs.append(ibh)
			bxs.append(bx.T)
			# bx is len by batch size
			bzs.append(bz.T)
		# if i > 3: break
		# n_known_x = np.vstack(bxs)
		known_h = np.vstack(bhs)
		known_ih = np.vstack(ibhs)
		# known_z = np.vstack(bzs)

		try:
			iprint('Trying to make a KDTree with leafsize 1,000')
			kd = BallTree(known_h, leaf_size=1000)
		except:
			iprint('That did not work, so trying again with leaf size 10,000')
			kd = BallTree(known_h, leaf_size=10000)

		try:
			iprint('Trying to make a KDTree with leafsize 1,000')
			ikd = BallTree(known_ih, leaf_size=1000)
		except:
			iprint('That did not work, so trying again with leaf size 10,000')
			ikd = BallTree(known_ih, leaf_size=10000)
		idec()
		iprint(
			'Done generating and caching explanatory representations of training data at {}. {} elapsed'.format(pu.now(), pu.now() - explanatory_rep_start_time))
	else:
		iprint('The num_related variable is set to 0, so not bothering to generate explanatory representations of training and validation sets')
		known_x = None
		known_y = None
		known_df = None
		kd = None
		ikd = None

	return known_x, known_y, known_df, kd, ikd


def generate_explanatory_representation(bx, bz, args, emb_func=None, padding_id=None, seq_pred_func=None):
	'''
	Generate vector representations of a set of examples using the appropriate method.
	:param bx:
	:param bz:
	:param args:
	:param emb_func:
	:param padding_id:
	:param seq_pred_func:
	:return:
	'''
	# if args.retrieval == 'final_h':
	# 	bh = h_final_func(bx, bz)[0]
	# 	ibh = h_final_func(bx, 1 - bz)[0]
	if args.retrieval == 'rationale_centroid':
		bh = generate_rationale_centroids(bx, bz, emb_func, padding_id)
		ibh = generate_rationale_centroids(bx, 1 - bz, emb_func, padding_id)
	elif args.retrieval == 'rationale_bigram_centroid':
		bh = generate_rationale_bigram_centroids(bx, bz, emb_func, padding_id)
		ibh = generate_rationale_bigram_centroids(bx, 1 - bz, emb_func, padding_id)
	elif args.retrieval == 'output_weighted_rationale_centroid':
		bh = generate_rationale_centroids(bx, bz, emb_func, padding_id, seq_pred_func=seq_pred_func)
		ibh = generate_rationale_centroids(bx, bz, emb_func, padding_id, seq_pred_func=seq_pred_func)
	else:
		raise Exception('Unknown retrieval type:{}'.format(args.retrieval))

	return bh, ibh


def batches_to_list(batches):
	lst = []
	for batch in batches:
		for item in batch:
			lst.append(item)
	return lst

def optimal_synthetic_z_func(x, embedding_layer):
	words = embedding_layer.map_to_words(x)
	rationale = [1 if word == 'bad' else 0 for word in words]
	return rationale_to_z_vector(rationale)

def synthetic_rationale_function(s):
	words = s.split()
	rationale = [1 if word == 'bad' else 0 for word in words]
	return rationale_to_z_vector(rationale,transpose=False)

def synthetic_with_neutral_rationale_function(s):
	words = s.split()
	rationale = [1 if word != 'neutral' else 0 for word in words]
	if minimize_synthetic_rationales:
		new_rationale = [0]*len(rationale)
		for i, zi in enumerate(rationale):
			if zi == 1:
				new_rationale[i] = 1
				break
		rationale = new_rationale

	return rationale_to_z_vector(rationale,transpose=False)

def rationale_to_z_vector(rationale, transpose=True):
	if transpose:
		return np.array([rationale]).astype(r.floatX).T
	else:
		return np.array(rationale).astype(r.floatX)


def optimal_synthetic_predictor(x,z,embedding_layer):
	py = 0.0
	ipy = 0.0
	for word, iz in zip(embedding_layer.map_to_words(x),z):
		if word == 'bad' and iz == 1:
			py = 1.0
		elif word == 'bad' and iz == 0:
			ipy = 1.0

	return np.array([[py]]).astype(r.floatX), \
	np.array([[ipy]]).astype(r.floatX)

#Unit testing

def assert_inverse_rationales(z1, z2):
	assert((z1 == (1-z2)).all())

def assert_model_rationalizing_correctly(pred_func):
	bx1 = np.array([[1, 2, 2, 2, 1, 2, 2]]).astype(np.int32).T
	bx2 = np.array([[1, 1, 2, 1, 1, 1, 2]]).astype(np.int32).T
	z =   np.array([[1, 0, 1, 0, 1, 0, 1]]).astype(r.floatX).T

	assert((pred_func(bx1, z)[0][0,0] == pred_func(bx2, z)[0][0,0]))




def print_evaluation(z, y, py, inverse_py, zero_py, eval_func, args, prefix='' , do_print = True):
	encoder_loss, generator_loss, prediction_loss, inverse_generator_prediction_loss, rationale_sparsity_loss, rationale_coherence_loss = eval_func(z, y, py, inverse_py, zero_py)

	if do_print:
		iprint(prefix, 'Chosen z: {}'.format(z.T))
		iprint(prefix, 'True y: {}'.format(y))
		iprint(prefix, 'Predicted y: {}'.format(py))
		iprint(prefix, 'Predicted inverse y: {}'.format(inverse_py))
		iprint(prefix, 'Predicted zero y: {}'.format(zero_py))
		iprint(prefix, 'Overall encoder loss: {}'.format(encoder_loss))
		iprint(prefix, '*Overall generator loss: {}'.format(generator_loss))
		iprint(prefix, 'Prediction loss: {}'.format(prediction_loss))
		iprint(prefix, 'inverse generator prediction loss: {} (*{}={})'.format(inverse_generator_prediction_loss,args.inverse_prediction_loss_weight, args.inverse_prediction_loss_weight*inverse_generator_prediction_loss))
		iprint(prefix, 'Rationale sparsity loss: {} (*{}={})'.format(rationale_sparsity_loss,args.z_sparsity_loss_weight, args.z_sparsity_loss_weight*rationale_sparsity_loss))
		iprint(prefix, 'Rationale coherence loss: {} (*{}={})'.format(rationale_coherence_loss,args.z, args.coherence_loss_weight*rationale_coherence_loss))

		print

	if np.isnan(py).any():
		iprint('Predicted a NaN while stress-testing model.')
		pass

	return encoder_loss,generator_loss





def create_or_load_embedding_layer(dir_path, size, text_srs, threads=8, existing_embedding_filepath=None):
	'''
	Run word2vec over a Pandas series of documents. Save all results to a directory. If files are already present in said directory, just load them in instead.
	:param dir_path:
	:param size:
	:param text_srs:
	:param threads:
	:return:
	'''

	if existing_embedding_filepath:
		iprint('Loading existing embedding file from {}'.format(existing_embedding_filepath))
		embedding_text_file_path = existing_embedding_filepath
	else:
		iprint('Creating word embeddings if necessary')
		text_file_path = os.path.join(dir_path, 'text.txt')
		phrase_file_path = os.path.join(dir_path, 'phrases.txt')
		embedding_bin_file_path = os.path.join(dir_path, 'embeddings.bin')
		embedding_text_file_path = os.path.join(dir_path, 'embeddings.txt.gz')

		text_file = 'text.txt'
		phrase_file = 'phrases.txt'
		embedding_bin_file = 'embeddings.bin'
		embedding_text_file = 'embeddings.txt.gz'

		# Python Word2vec package can't deal with filepaths of more than 60 characters
		# for some stupid reason, so we need to create these files wherever the script is running and then move them afterward
		if not os.path.isfile(embedding_text_file_path):
			iprint('Creating raw text file')
			text_srs.to_csv(text_file, header=False, index=False)
			iprint('Creating phrase file')
			word2vec.word2phrase(text_file, phrase_file, verbose=False)
			iprint('Creating word vector binary file')
			word2vec.word2vec(phrase_file, embedding_bin_file, size=size, verbose=False, threads=8)


			iprint('Converting word vector binary file to text file')
			model = word2vec.load(embedding_bin_file)
			with gzip.open(embedding_text_file, 'wb') as of:
				for i, v in enumerate(model.vocab):
					of.write(v.encode('utf-8'))
					of.write(' ')
					of.write(' '.join([str(x) for x in model.vectors[i]]))
					of.write('\n')

			# Move all generated files to the correct location
			iprint('Moving files to the correct location')
			os.rename(text_file, text_file_path)
			os.rename(phrase_file, phrase_file_path)
			os.rename(embedding_bin_file, embedding_bin_file_path)
			os.rename(embedding_text_file, embedding_text_file_path)
		else:
			iprint('Embedding file {} already exists, so not re-creating it'.format(embedding_text_file_path))

	embedding_layer = myio.create_embedding_layer(embedding_text_file_path)
	return embedding_layer


def initialize_output_json_object():
	output = {}
	output['platforms'] = []
	output['dimensions'] = []
	output['comments'] = []
	output['ratings'] = []
	output['results'] = []
	output['relatedexamples'] = []
	output['predictivemodels'] = []
	output['relationtype'] = []
	return output




def add_rationales_to_df(df, rationales, sample = 100):
	'''

	:param df: dataframe of a dataset. Must have a
	:param rationales: either a filename or a function. If a filename, it should be a csv with columns platform_comment_id and rationale, where the rationale is a json list of 1s and 0s. If a function it should map a text comment to a rationale which is a numpy vector
	:param sample: if this is not None, then sample this percentage of the original df and generate rationales for that sample only.
	:return:
	'''

	if type(rationales) == str:
		iprint('Loading rationales from file {}'.format(rationales))
		rationale_df = pd.read_csv(rationales)
		rationale_df['rationale'] = rationale_df['rationale'].apply(lambda x:rationale_to_z_vector(json.loads(x), False))
		df = df.merge(rationale_df, on='platform_comment_id',how='left')
		matched = df[df['rationale'].notnull()].shape[0]
		if matched != rationale_df.shape[0]:
			iprint('ERROR: Only {} of {} rationales in rationale file could be matched to items in dataframe.'.format(matched, rationale_df.shape[0]))
		return df
	elif callable(rationales):
		iprint('Generating synthetic rationales using function: {}'.format(rationales))
		if minimize_synthetic_rationales:
			iprint('Taking only the first token of each generated synthetic rationale')
		df['rationale'] = df['text'].sample(n=sample,random_state=seed).apply(rationales)
		return df
	else:
		raise Exception("Type {} is not supported for rationales argument".format(type(rationales)))

class CallableEncoder(json.JSONEncoder):
	def default(self, o):
		if callable(o):
			return str(0)
		else:
			return json.JSONEncoder.default(self, o)




def read_multiple_csvs_into_one_df(file_value):
	'''
	Load multiple CSV files into one dataframe. Will give an error if they don't all have the same columns
	:param file_value:
	:return:
	'''
	if type(file_value) == str:
		filenames = [file_value]
	else:
		filenames = file_value

	dfs_unlabeled = []

	for filename in filenames:
		iprint('Reading in unlabeled dataset at {}'.format(filename))
		sub_df_unlabeled = pd.read_csv(filename)
		dfs_unlabeled.append(sub_df_unlabeled)

	df_unlabeled = pd.concat(dfs_unlabeled, axis=0)

	return df_unlabeled


class Wrapper():
	def __init__(self,v):
		self.v = v


class RationaleException(Exception):
	def __init__(self, *args, **kwargs):
		Exception.__init__(self, *args, **kwargs)

if __name__ == '__main__':
	main()