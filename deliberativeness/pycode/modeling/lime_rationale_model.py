
import gzip
import os
import pickle
import traceback

import numpy as np
import sklearn.metrics as mt
import theano
from lime import lime_text
from sklearn.pipeline import make_pipeline

import misc.s_options as options
from rationale import myio
import processing.putil as pu
from modeling.carton_rationale import Model
from modeling.carton_rationale import ModelObjectiveEvaluation
from nn import EmbeddingLayer
from processing.putil import iprint, idec, iinc


class LIMEModel(Model):
	def __init__(self, *args, **kwargs):
		Model.__init__(self, *args, **kwargs)

		assert(isinstance(self.args, options.Arguments))
		lime_updates = {
			'fix_rationale_value':1
		}
		iprint('Modifying model args to accommodate the fact that we are using LIME for rationales: {}'.format(lime_updates))
		options.load_arguments(lime_updates, self.args)

		self.threshold = 0.001

		self.rationale_cache = {}


	def ready(self):
		Model.ready(self)

		self.default_value = pu.sigmoid(float(self.encoder.output_layer.b.get_value()))
		self.vectorizor = self.LimeVectorizor(self.embedding_layer)
		self.prediction_function = self.LimeFunction(self.simple_prediction_function, self.default_value, self.args, self.padding_id)
		self.pipeline = make_pipeline(self.vectorizor, self.prediction_function)
		self.explainer = lime_text.LimeTextExplainer(bow=False, split_expression=' ', feature_selection='none')

	def train(self, train, dev, test, encoder_only=False, dev_rationales=None):

		iprint('Training original Model model')

		already_trained=self.epochs
		iinc()
		training_result = Model.train(self, train, dev, test, encoder_only, dev_rationales)
		idec()


		# if already_trained == 0 or self.epochs > already_trained: #if any new training happened, or the model hadn't been trained before, tune the threshold
		iprint('Tuning inclusion threshold to maximize F1 on development set rationales')
		iprint('Predicting rationale for each dev rationale')
		nonnull_dev_rationales = []
		predicted_dev_rationale_probs = []
		iinc()
		for dev_rationale, dev_xi in zip(dev_rationales, dev[0]):
			if not np.any(np.isnan(dev_rationale)):
				nonnull_dev_rationales.append(dev_rationale)
				probs = self.explain_item_with_lime(dev_xi)
				predicted_dev_rationale_probs.append(probs)
				assert (len(dev_rationale)== len(probs))

				if len(nonnull_dev_rationales) % 25 == 0:
					iprint('{}...'.format(len(nonnull_dev_rationales)))
					# iprint('Ending early for speed purposes. Turn this off.') #TODO turn this off
					# break
		idec()

		combined_dev_rationale = np.hstack(nonnull_dev_rationales)
		combined_predicted_probs = np.hstack(predicted_dev_rationale_probs)

		# assert(combined_dev_rationale.shape[0] == combined_predicted_probs.shape[0])

		num_intervals = 50
		unique_probs = np.sort(np.unique(np.abs(combined_predicted_probs)))
		interval_size = unique_probs.shape[0] / num_intervals
		threshold_values = unique_probs[interval_size::interval_size]
		iprint('Investigating {} possible values for threshold between {} and {}'.format(num_intervals, threshold_values[0], threshold_values[-1]))

		best_threshold_value = None
		best_f1 = 0

		iinc()
		for threshold_value in threshold_values:
			threshold_combined_rationale = (combined_predicted_probs >= threshold_value).astype(int)
			threshold_f1 = mt.f1_score(combined_dev_rationale, threshold_combined_rationale)
			threshold_precision = mt.precision_score(combined_dev_rationale, threshold_combined_rationale)
			threshold_recall = mt.recall_score(combined_dev_rationale, threshold_combined_rationale)
			iprint('{:.6f}:\tp={:.3f}\tr={:.3f}\tf1={:.3f}'.format(threshold_value, threshold_precision, threshold_recall, threshold_f1))
			if threshold_f1 > best_f1:
				best_f1 = threshold_f1
				best_threshold_value = threshold_value

		idec()

		iprint('Best f1 found at threshold value {}: {}'.format(best_threshold_value, best_f1))
		self.threshold = best_threshold_value

		iprint('Saving model with tuned threshold to: {}'.format(self.args.save_model))
		self.save_model(self.args.save_model)
			# self.load_model(self.args.save_model)
			# pass
		# else:
		# 	iprint('Not tuning lime weight inclusion threshold, because the model had been trained for {} epochs before the train method was called and has now been trained for {}'.format(already_trained, self.epochs))

		return training_result


	# def vectorize(self, text):
	# 	return np.asarray(self.embedding_layer.map_to_ids(text.split())).reshape((-1,1))

	class LimeVectorizor():

		def __init__(self, embedding_layer = None, *args, **kwargs):
			self.embedding_layer = embedding_layer

		def fit(self, text):
			pass

		def transform(self, texts):
			return [np.asarray(self.embedding_layer.map_to_ids(text.split())) for text in texts]
			# return [np.asarray(self.embedding_layer.map_to_ids(text.split())).reshape((1, -1)) for text in texts]


	class LimeFunction():
		def __init__(self, theano_function, default_value, args, padding_id):
			self.function = theano_function
			self.default_value = default_value
			self.args = args
			self.padding_id = padding_id

		def fit(self,*args, **kwargs):
			'''
			Just to comply with the sklearn pipeline API
			:param args:
			:param kwargs:
			:return:
			'''
			pass

		def predict_proba(self, x_lst):
			'''
			Return predictions for a list of input instances in a way that conforms with the sklearn API
			:param x_lst:
			:return:
			'''

			r = [None]*len(x_lst)
			x_batches, _, i_batches = myio.create_batches(x_lst, None, self.args.batch, self.padding_id, return_indices=True)

			for x_batch, i_batch in zip(x_batches, i_batches):
				bpy = self.function(x_batch)[0]
				for py, i in zip(bpy, i_batch):
					r[int(i)] = [1-py,py]


			assert(np.all(r))
			# for x in x_lst:
				# py = float(self.function(x.T)[0]) if x.shape[0] > 0 else self.default_value
				# r.append([1-py, py])

			return np.asarray(r)


	def lime_itemwise_prediction_function(self, bx, by, threshold = None, btz=None):

		# if threshold is None:
		# 	threshold = self.threshold
		#
		# #TODO wayyyy too slow. Must add batching to  LimeFunction
		# default_value = pu.sigmoid(float(self.encoder.output_layer.b.get_value()))
		# vectorizor = self.LimeVectorizor(self.embedding_layer)
		# prediction_function = self.LimeFunction(self.simple_prediction_function, default_value, self.args, self.padding_id)
		# pipeline = make_pipeline(vectorizor, prediction_function)
		# explainer = lime_text.LimeTextExplainer(bow=False, split_expression = ' ', feature_selection='none')
		# assert (isinstance(self.embedding_layer, EmbeddingLayer))
		#
		# #For each (padded) xi in bx:
		# 	#unpad it
		# 	#Generate a "dataset" of versions of that xi missing random words
		#
		#
		#
		# bzis = []
		# for i, bxi in enumerate(bx.T):
		# 	# tick()
		# 	iprint(i)
		# 	unpadded_bxi = pu.unpad(bxi, self.padding_id)
		# 	words = self.embedding_layer.map_to_words(unpadded_bxi.T)
		# 	text = ' '.join(words)
		# 	exp = explainer.explain_instance(text, pipeline.predict_proba,sort=False)
		# 	# expdict = dict(exp.as_list())
		# 	bzi = np.matrix([1 if np.abs(weight) >= threshold else 0 for word, weight in exp.as_list()]).T
		# 	bzi = np.pad(bzi, (bxi.shape[0]-bzi.shape[0],0), 'constant')
		# 	bzis.append(bzi)
		# 	# tock()
		#
		#
		# bz = np.hstack(bzis)

		bz, _, _ = self.lime_rationale_function(bx, btz=btz)


		py = self.simple_prediction_function(bx)[0]
		result = ModelObjectiveEvaluation({'py':py, 'pz':bz})
		# result['pz'] = bz

		return result


	def lime_rationale_function(self, bx, threshold = None, btz=None):
		if threshold is None:
			threshold = self.threshold

		# TODO wayyyy too slow. Must add batching to  LimeFunction
		# default_value = pu.sigmoid(float(self.encoder.output_layer.b.get_value()))
		# vectorizor = self.LimeVectorizor(self.embedding_layer)
		# prediction_function = self.LimeFunction(self.simple_prediction_function, default_value, self.args, self.padding_id)
		# pipeline = make_pipeline(vectorizor, prediction_function)
		# explainer = lime_text.LimeTextExplainer(bow=False, split_expression=' ', feature_selection='none')
		assert (isinstance(self.embedding_layer, EmbeddingLayer))

		# For each (padded) xi in bx:
		# unpad it
		# Generate a "dataset" of versions of that xi missing random words



		pzs = []
		ct =0
		numreal = np.sum(btz)
		iinc()
		for i, bxi in enumerate(bx.T):
			# tick()
			bxistr = str(bxi)


			if btz is not None and btz[i] == False:
				bpzi = np.ones_like(bxi)
			else:
				ct += 1
				if ct % 10 == 0:
					iprint('{}/{}...'.format(ct,numreal))
				bpzi = self.explain_item_with_lime(bxi)
			bpzi = bpzi.reshape((bpzi.shape[0], 1)).astype(theano.config.floatX)
			pzs.append(bpzi)
			# if bxistr in self.rationale_cache:
			# 	pzs.append(self.rationale_cache[bxistr])
			# else:
			#
			# 	# iprint(i)
			# 	unpadded_bxi = pu.unpad(bxi, self.padding_id)
			# 	words = self.embedding_layer.map_to_words(unpadded_bxi.T)
			# 	text = ' '.join(words)
			# 	exp = explainer.explain_instance(text, pipeline.predict_proba, sort=False)
			# 	# expdict = dict(exp.as_list())
			# 	# bzi = np.matrix([1 if np.abs(weight) >= threshold else 0 for word, weight in exp.as_list()]).T
			# 	bpzi = np.matrix([weight for word, weight in exp.as_list]).T
			# 	bpzi = np.pad(bpzi, (bxi.shape[0] - bpzi.shape[0], 0), 'constant')
			# 	pzs.append(bpzi)
			# 	self.rationale_cache[bxistr] = bpzi
		# tock()

		idec()
		probs = np.hstack(pzs)

		z = (probs > threshold).astype(theano.config.floatX)
		inverse_z = 1-z


		return z, inverse_z, probs


	def explain_item_with_lime(self, bxi):
		'''
		Explains a vector of token IDs by passing them to simple_prediction_function and then running LIME over that

		Returns probs
		:param bxi:
		:return:
		'''
		bxistr = str(bxi)
		if bxistr in self.rationale_cache:
			return self.rationale_cache[bxistr]
		else:
			# iprint(i)
			unpadded_bxi = pu.unpad(bxi, self.padding_id)
			words = self.embedding_layer.map_to_words(unpadded_bxi.T)
			try:
				text = ' '.join(words).decode('utf-8')
			except:
				text = ' '.join(words)
			exp = self.explainer.explain_instance(text, self.pipeline.predict_proba, sort=False)
			# expdict = dict(exp.as_list())
			# bzi = np.matrix([1 if np.abs(weight) >= threshold else 0 for word, weight in exp.as_list()]).T
			bpzi = np.asarray([weight for word, weight in exp.as_list()])
			bpzi = np.pad(bpzi, (bxi.shape[0] - bpzi.shape[0], 0), 'constant')

			self.rationale_cache[bxistr] = bpzi
			return bpzi

	# def lime_evaluation_prediction_function(self, bx, bz, by):
	# 	pass

	def compile_functions(self):
		Model.compile_functions(self)



		self.simple_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=[self.encoder.py],
			updates=self.generator.sample_updates
		))

		self.rationale_function = self.lime_rationale_function

		self.itemwise_prediction_function = self.lime_itemwise_prediction_function

		# self.evaluation_prediction_function = self.lime_evaluation_prediction_function

	def save_model(self, path):
		# append file suffix
		if not path.endswith(".pkl.gz"):
			if path.endswith(".pkl"):
				path += ".gz"
			else:
				path += ".pkl.gz"

		# output to path
		if not self.args.split_encoder:
			with gzip.open(path, "wb") as fout:
				pickle.dump(
					([x.get_value() for x in self.encoder.params],  # encoder
					 [x.get_value() for x in self.generator.params],  # generator
					 self.nclasses,
					 self.args,  # training configuration
					 self.epochs,
					 self.useful_epochs,
					 self.encoder_epochs,
					 self.threshold,
					 self.rationale_cache
					 ),
					fout,
					protocol=pickle.HIGHEST_PROTOCOL
				)
		else:
			with gzip.open(path, "wb") as fout:
				pickle.dump(
					([x.get_value() for x in self.encoder.params],  # encoder
					 [x.get_value() for x in self.encoder.secondary_params],  # encoder
					 [x.get_value() for x in self.generator.params],  # generator
					 self.nclasses,
					 self.args,  # training configuration
					 self.epochs,
					 self.useful_epochs,
					 self.encoder_epochs,
					 self.threshold,
					 self.rationale_cache
					 ),
					fout,
					protocol=pickle.HIGHEST_PROTOCOL
				)
		pass

	def load_model(self, path, load_args=False):
		if not os.path.exists(path):
			if path.endswith(".pkl"):
				path += ".gz"
			else:
				path += ".pkl.gz"

		# if self.args.split_encoder:
		with gzip.open(path, "rb") as fin:
			eparams, inverse_eparams, gparams, nclasses, args, epoch, useful_epoch, encoder_epochs, threshold, rationale_cache = pickle.load(fin)
		# else:
		# 	with gzip.open(path, "rb") as fin:
		# 		eparams, gparams, nclasses, args, epoch, useful_epoch, encoder_epochs = pickle.load(fin)

		# construct model/network using saved configuration
		# self.args = args



		self.nclasses = nclasses
		self.epochs = epoch
		self.useful_epochs = useful_epoch
		self.encoder_epochs = encoder_epochs
		self.threshold = threshold
		self.rationale_cache = rationale_cache
		if load_args:
			self.args = args

		self.ready()

		for x, v in zip(self.encoder.params, eparams):
			try:
				x.set_value(v)
			except Exception as ex:
				traceback.print_exc()
				iprint('Param shape: {}; Input shape: {}'.format(x.get_value().shape, v.shape))
				iprint()

		if self.args.split_encoder:
			for x, v in zip(self.encoder.secondary_params, inverse_eparams):
				x.set_value(v)

		for x, v in zip(self.generator.params, gparams):
			x.set_value(v)