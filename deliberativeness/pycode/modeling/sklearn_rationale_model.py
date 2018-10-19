
from copy import deepcopy

import numpy as np
import sklearn.metrics as mt
from sklearn.linear_model import LogisticRegression

import misc.s_options as options
from modeling.carton_rationale import ModelObjectiveEvaluation
from processing.putil import iprint, idec, iinc, tick, tock


class L2RegressionModel():
	def __init__(self, args=None, embedding_layer=None, nclasses=-1):

		self.args = args
		# assert(isinstance(self.args,options.Arguments))
		self.embedding_layer = embedding_layer
		self.padding_id = self.embedding_layer.vocab_map["<padding>"]

		self.threshold = 0.001
		self.trained = False

		if nclasses != 1:
			raise Exception('More than 1 class is not supported for l2 regression model')

	def ready(self):
		iprint('Readying simple sklearn LR model')

	def train(self, train, dev, test, dev_rationales=None, fit_threshold = True):
		assert(isinstance(self.args,options.Arguments))
		iprint('Training scikit-learn model')
		model = self.model = LogisticRegression(penalty='l1',C=self.args.sklearn_l1_C)

		seq_train_x, train_y = train
		seq_dev_x, dev_y = dev

		vec_train_x = self.to_bag_of_words(seq_train_x)
		vec_dev_x = self.to_bag_of_words(seq_dev_x)

		bin_train_y = self.binarize(train_y)

		tick('Fitting model')
		model.fit(vec_train_x, bin_train_y)
		tock('Done fitting model')

		if fit_threshold and dev_rationales is not None:
			tick('Finding a coefficient threshold that maximizes F1 with respect to dev set rationales')

			rationale_indices = [i for i, r in enumerate(dev_rationales) if not np.any(np.isnan(r))]
			rationale_seq_dev_x = seq_dev_x[rationale_indices]
			nonnull_dev_rationales = dev_rationales[rationale_indices]
			combined_dev_rationale = np.hstack(nonnull_dev_rationales)

			# min_val = np.min(np.abs(model.coef_))
			# max_val = np.max(np.abs(model.coef_))
			num_intervals = 50
			unique_coefficients = np.sort(np.unique(np.abs(model.coef_)))
			interval_size = unique_coefficients.shape[0]/num_intervals
			threshold_values = unique_coefficients[interval_size::interval_size]
			# threshold_values = unique_coefficients[0::interval_size]


			iprint('Exploring {} possible values for coefficient cutoff, maximizing F1 on development set rationales. '.format(num_intervals))
			_, _, probs = self.rationale_function(rationale_seq_dev_x)

			best_f1 = 0
			best_threshold_value = None
			iinc()
			for threshold_value in threshold_values:
				threshold_rationale_dev_pz, _, _ = self.rationale_function(rationale_seq_dev_x, probs=probs, threshold=threshold_value)
				combined_threshold_pz = np.hstack(threshold_rationale_dev_pz)
				threshold_f1 = mt.f1_score(combined_dev_rationale, combined_threshold_pz)
				threshold_recall = mt.recall_score(combined_dev_rationale, combined_threshold_pz)
				threshold_precision = mt.precision_score(combined_dev_rationale, combined_threshold_pz)

				iprint('{:.6f}:\tp={:.3f}\tr={:.3f}\tf1={:.3f}'.format(threshold_value, threshold_precision, threshold_recall, threshold_f1))
				if threshold_f1 > best_f1:
					best_f1 = threshold_f1
					best_threshold_value = threshold_value

			idec()
			tock('Done searching. Best combined dev rationale f1 {} achieved at threshold value {}'.format(best_f1, best_threshold_value))
			self.threshold = best_threshold_value

		iprint('Done with model training')
		return []



	def itemwise_prediction_function(self, bx,by):

		result = ModelObjectiveEvaluation()
		vec_bx = self.to_bag_of_words(bx, transposed=True, ignore_padding=True)

		bpz, inverse_bpz, probs = self.rationale_function(bx)
		result['pz'] = bpz

		# bpy = self.model.predict_proba(vec_bx)[:,1]
		# result['py'] = bpy

		result.update(self.evaluation_prediction_function(bx,bpz,by))

		return result


	def evaluation_prediction_function(self, bx,bz,by):

		bz = bz.astype(int)
		# bx_z = bx*bz #I decided I didn't want to actually incorporate the mask into the prediction.
		bx_z = bx

		vec_bx_z = self.to_bag_of_words(bx_z, transposed=True, ignore_padding = True)
		py = self.model.predict_proba(vec_bx_z)[:,1:2]

		return ModelObjectiveEvaluation({'py':py})


	def rationale_function(self, x, probs = None, threshold = None):
		'''
		Takes a datastructure of token IDs, looks them up within the parameter array of the model, and
		chooses whether to include or exclude them from the rationale. Tries to be flexible about the structure of x,
		returning the rationale in the same form as x
		:param x: can be a numpy array, or a sequence of numpy arrays
		:param probs:
		:param threshold:
		:return:
		'''
		if threshold is None:
			threshold = self.threshold


		if probs is None:
			probs = deepcopy(x)
			fillprobs = True
		else:
			fillprobs = False
		pz = deepcopy(x)
		inverse_pz = deepcopy(x)
		for i,xi in enumerate(x):
			for j, xij in enumerate(xi):
				if fillprobs:
					probs[i][j] = self.model.coef_[0][xij]

				#Todo this is only capturing positive coefficients right now
				pz[i][j] = int(probs[i][j] >= threshold)
				inverse_pz[i][j] = 1-pz[i][j]
		# pz = (probs >= threshold).astype(int)
		# inverse_pz = 1-pz

		return pz, inverse_pz, probs

	def to_bag_of_words(self, seq_x, transposed = False, ignore_padding = False):
		'''

		:param x: a sequence of vectors of integers, where each integer represents a token in the vocabulary of the embedding_layer
		:return:
		'''

		if transposed:
			seq_x = seq_x.T

		vec_x = np.zeros((len(seq_x), self.embedding_layer.n_V))
		for i, xi in enumerate(seq_x):
			for xij in xi:
				if ignore_padding and xij == self.padding_id:
					pass
				else:
					vec_x[i][xij] += 1

		return vec_x



		pass

	def save_model(self, path):
		iprint('Model saving not implemented yet')

	def binarize(self, y):
		return (y >= 0.5).astype(float)