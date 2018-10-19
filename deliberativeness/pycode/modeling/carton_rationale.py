
import cPickle as pickle
import gzip
import os
import time
import traceback
from collections import OrderedDict

import numpy as np
import pandas as pd
import sklearn.metrics as mt
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

import misc.s_options as options
from rationale import myio
import processing.putil as pu
import sam_util as su
from rationale.extended_layers import ExtRCNN, ExtLSTM, ZLayer
from nn import Layer, LSTM, RCNN, apply_dropout, EmbeddingLayer
from nn import create_optimization_updates, get_activation_by_name, sigmoid
from processing.putil import iprint, iinc, idec, tick, tock, mean_dict_list

floatX = theano.config.floatX

class Generator(object):
	def __init__(self, args, embedding_layer, nclasses):
		self.args = args
		self.embedding_layer = embedding_layer
		self.nclasses = nclasses

	def ready(self):
		embedding_layer = self.embedding_layer
		args = self.args
		assert(isinstance(args, options.Arguments))
		padding_id = embedding_layer.vocab_map["<padding>"]

		dropout = self.dropout = theano.shared(
			np.float64(0).astype(theano.config.floatX)
		)

		# len*batch
		x = self.x = T.imatrix()
		self.batch_width = x.shape[0]

		embs = embedding_layer.forward(x.ravel())
		# len*batch*n_e
		n_e = embedding_layer.n_d
		embs = embs.reshape((x.shape[0], x.shape[1], n_e))
		self.pre_dropout_embs = embs
		embs = apply_dropout(embs, dropout)

		self.word_embs = embs

		reversed_embs = embs[::-1]

		# len * batch
		masks = self.masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

		self.padding_fraction = T.mean(1-masks)

		# x = su.sprint(x, 'x upon input to generator')


		n_d = args.hidden_dimension
		activation = get_activation_by_name(args.activation)

		layers = self.layers = []

		if args.generator_architecture.lower() == 'rnn':
			iprint('Using a recurrent neural net architecture for the generator')
			layer_type = args.layer.lower()

			# Generator is always a 2 layer rnn so that it can be bidirectional
			for i in xrange(2):
				if layer_type == "rcnn":
					l = RCNN(
						n_in=n_e,
						n_out=n_d,
						activation=activation,
						order=args.order,
						name='generator_layer_{}'.format(i)
					)
				elif layer_type == "lstm":
					l = LSTM(
						n_in=n_e,
						n_out=n_d,
						activation=activation,
						name = 'generator_layer_{}'.format(i)
					)
				layers.append(l)



			# (len*batch)*n_e

			# len*batch*n_d
			h1 = self.h1 =  layers[0].forward_all(embs,return_c = args.return_c)
			h2 = self.h2 = layers[1].forward_all(reversed_embs, return_c = args.return_c)

			try:
				if args.bidirectional_generator:
					iprint('Making generator bidirectional')
					h_final = T.concatenate([h1, h2[::-1]], axis=2)
					size = n_d * 2
				else:
					iprint('Making generator unidirectional')
					h_final = h1
					size = n_d
			except:
				iprint('Making generator bidirectional')
				h_final = T.concatenate([h1, h2[::-1]], axis=2)
				size = n_d * 2

			if args.return_c:
				iprint("Generator RNN outputs its hidden state, so double size of internal representation for subsequent layers")
				size *= 2

			self.pre_dropout_h_final = h_final
			h_final = apply_dropout(h_final, dropout)

			# h_final = su.sprint(h_final, 'generator h_final')

			self.h_final = h_final


			# len * batch * 2n_d

			#If we choose to use dependent version of the model, then output layer needs to be a recurrent layer where each zi is dependent on the sampled value
			#of zi-1




			if not args.hard_attention:
				iprint('Using a soft attention mechanism')
				output_layer = self.output_layer = Layer(
					n_in=size,
					n_out=1,
					activation=sigmoid,
					name='generator_output_layer'
				)
				probs = output_layer.forward(h_final)
				# probs = su.sprint(probs, 'Soft attention probs')
				probs = self.probs =  probs.reshape(x.shape)
				z_sample = self.z_sample = probs
				mle_probs = self.mle_probs = output_layer.forward(h_final).reshape(x.shape)
				mle_z = self.mle_z =  T.cast(mle_probs >= 0.5, theano.config.floatX)
				# z_sample = su.sprint(z_sample, 'Soft attention z_sample')
				self.sample_updates = []
				self.mle_sample_updates = []
			else:
				iprint('Using a hard attention mechanism')
				if not args.dependent:
					iprint('Using independent version of generator')
					output_layer = self.output_layer = Layer(
						n_in=size,
						n_out=1,
						activation=sigmoid,
						name='generator_output_layer'
					)
					probs = output_layer.forward(h_final)
					# probs = su.sprint(probs, 'Hard attention independent probs')


					probs = self.probs = probs.reshape(x.shape)
					self.MRG_rng = MRG_RandomStreams()
					z_sample = T.cast(self.MRG_rng.binomial(size=probs.shape, p=probs), theano.config.floatX)
					z_sample = self.z_sample = theano.gradient.disconnected_grad(z_sample)

					mle_probs = self.mle_probs = probs

					mle_z = self.mle_z = T.cast(probs >= 0.5, theano.config.floatX)
					# z_sample = su.sprint(z_sample, 'Hard attention independent z_sample')

					# z_sample = self.MRG_rng.binomial(size=probs.shape, p=probs)

					self.sample_updates = []  # This is only really a thing for the dependent version of the algorithm
					self.mle_sample_updates = []

				else:
					iprint('Using dependent version of generator')

					if not args.add_counter:
						size_increase=1
					else:
						size_increase=2

					if args.layer == 'lstm':
						output_layer = self.output_layer = ZLayer(
							n_in=size,
							n_hidden=args.hidden_dimension2,
							activation=activation,
							rlayer = LSTM(n_in=( size +size_increase) ,n_out=args.hidden_dimension2 ,activation=activation)
						)
					else:
						output_layer = self.output_layer = ZLayer(
							n_in=size,
							n_hidden=args.hidden_dimension2,
							activation=activation,
							rlayer=RCNN((size+size_increase), args.hidden_dimension2, activation=activation, order=2, name='generator_dependent_recurrent_layer'),
							name='generator_dependent_layer'
						)

					if args.add_counter:
						iprint('Adding a counter memory cell to the generator sequence model')
						z_sample, probs, self.cz, h, sample_updates = output_layer.sample_all(h_final, add_counter=True)
						mle_z, mle_probs, self.mle_cz ,_, mle_sample_updates = output_layer.sample_all(h_final, mle=True, add_counter=True)
					else:
						z_sample, probs, h, sample_updates = output_layer.sample_all(h_final)
						mle_z, mle_probs,_, mle_sample_updates = output_layer.sample_all(h_final, mle=True)


					# z_sample = self.z_sample = theano.gradient.disconnected_grad(z_sample)
					self.z_sample = z_sample
					self.probs = probs
					self.sample_updates=sample_updates

					# mle_z = self.mle_z = theano.gradient.disconnected_grad(mle_z)
					self.mle_z = mle_z
					self.mle_probs = mle_probs
					self.mle_sample_updates = mle_sample_updates

					# z_sample, h, sample_updates = output_layer.sample_all(h_final)
					# mle_z, _, mle_sample_updates = output_layer.sample_all(h_final, mle=True)
					#
					# z_sample = self.z_sample = theano.gradient.disconnected_grad(z_sample)
					# probs = self.probs = output_layer.forward_all(h_final, z_sample)
					# self.sample_updates = sample_updates
					#
					# mle_z = self.mle_z = theano.gradient.disconnected_grad(mle_z)
					# self.mle_sample_updates = mle_sample_updates


				# this calculates the (log) probability of the predicted rationale z based on the current model
				logpz = self.logpz =  -T.nnet.binary_crossentropy(probs.clip(0.01,0.99), z_sample) * masks
				# logpz.name = 'logpz'
				# logpz = self.logpz = logpz.reshape(x.shape)
				# probs = probs.reshape(x.shape)

			# logpz = logpz
				# self.logpz = logpz


		elif args.generator_architecture.lower() == 'sigmoid':
			iprint('Using a single sigmoid layer applied to each token (logistic regression) for generator')
			output_layer = self.output_layer = Layer(
				n_in=n_e,
				n_out=1,
				activation=sigmoid,
				name='generator_output_layer'
			)
			probs = output_layer.forward(embs)
			probs = self.probs = probs.reshape(x.shape)

			if args.hard_attention:
				iprint('Using hard attention mechanism')
				#Copied from above
				self.MRG_rng = MRG_RandomStreams()
				z_sample = T.cast(self.MRG_rng.binomial(size=probs.shape, p=probs), theano.config.floatX)
				z_sample = self.z_sample = theano.gradient.disconnected_grad(z_sample)

				mle_probs = self.mle_probs = probs

				mle_z = self.mle_z = T.cast(probs >= 0.5, theano.config.floatX)
				# z_sample = su.sprint(z_sample, 'Hard attention independent z_sample')

				# z_sample = self.MRG_rng.binomial(size=probs.shape, p=probs)

				self.sample_updates = []  # This is only really a thing for the dependent version of the algorithm
				self.mle_sample_updates = []
				logpz = self.logpz = -T.nnet.binary_crossentropy(probs.clip(0.01, 0.99), z_sample) * masks

			else:
				raise Exception('Soft attention is not implemented for simple sigmoid layer generator')


		if args.fix_rationale_value == 1 or args.fix_rationale_value == 0:
			iprint('Fixing rationale values at {}'.format(args.fix_rationale_value))
			# If we've chosen not to use rationales, then the generator should trivially produce all 1s
			# print 'Setting generator to always just predict ones'
			if args.fix_rationale_value == 1:
				z_sample = T.ones_like(x, dtype=theano.config.floatX)
				mle_z = T.ones_like(x, dtype=theano.config.floatX)
				probs =0.99*T.ones_like(x, dtype=theano.config.floatX)
			else:
				z_sample = T.zeros_like(x, dtype=theano.config.floatX)
				mle_z = T.zeros_like(x, dtype=theano.config.floatX)
				probs = 0.01*T.ones_like(x, dtype=theano.config.floatX)

			mle_z = self.mle_z = theano.gradient.disconnected_grad(mle_z)
			z_sample = self.z_sample = theano.gradient.disconnected_grad(z_sample)
			probs = self.probs =  theano.gradient.disconnected_grad(probs)
		elif args.fix_rationale_value is not None:
			raise Exception('{} is not an allowed value for rationale to be set to'.format(args.fix_rationale_value))

		params = self.params = []
		for l in layers + [output_layer]:
			for p in l.params:
				params.append(p)
		nparams = sum(len(p.get_value(borrow=True).ravel()) for p in params)
		paramsizes = [(p.name, len(p.get_value(borrow=True).ravel())) for p in params]
		iprint("Generator params: total {}".format(nparams))
		# iprint('\n'.join(['\t{}: {}'.format(name, num) for name, num in paramsizes]))


		inverse_z_sample = self.inverse_z_sample = 1 - z_sample

		rng = RandomStreams(seed=args.rng_seed)  # Use a seed to help reproduce results

		confusion_prob = self.confusion_prob = theano.shared(np.float64(0.5).astype(theano.config.floatX))
		confusion_indices = self.confusion_indices = rng.binomial(size=(z_sample.shape[1],), p=confusion_prob)

		perm = rng.permutation(n=inverse_z_sample.shape[1])
		fully_shuffled_inverse_z_sample = inverse_z_sample[:,perm]
		shuffled_inverse_z_sample = self.shuffled_inverse_z_sample = T.cast(inverse_z_sample*(1-confusion_indices) + fully_shuffled_inverse_z_sample*confusion_indices, theano.config.floatX)

		fully_flipped_inverse_z_sample = inverse_z_sample[:,::-1]
		flipped_inverse_z_sample = self.flipped_inverse_z_sample = T.cast(inverse_z_sample*(1-confusion_indices) + fully_flipped_inverse_z_sample*confusion_indices, theano.config.floatX)


		fully_shuffled_z_sample = z_sample[:,perm]
		shuffled_z_sample = self.shuffled_z_sample = T.cast(z_sample*(1-confusion_indices) + fully_shuffled_z_sample*confusion_indices, theano.config.floatX)

		fully_flipped_z_sample = z_sample[:,::-1]
		flipped_z_sample = self.flipped_z_sample = T.cast(z_sample*(1-confusion_indices) + fully_flipped_z_sample*confusion_indices, theano.config.floatX)

		# inverse_z_sample = z_sample
		zero_z= self.zero_z = T.zeros_like(z_sample)


		#I think these are unecessary
		# logpz = - T.nnet.binary_crossentropy(probs, z_sample) * masks
		# logpz = logpz.reshape(x.shape)
		# probs  = probs.reshape(x.shape)



		#Make sure that padding is always obscured no matter what
		# probs = probs * masks

		#These may not be necessary
		#TODO figure out a reasonable way of dealing with masks
		# z_sample = z_sample * masks
		# inverse_z_sample = inverse_z_sample * masks
		# mle_z = mle_z * masks
		# inverse_mle_z = inverse_mle_z* masks


		#Create instance variables
		# self.z_sample = z_sample

		# batch
		self.zsum = T.sum(z_sample, axis=0, dtype=theano.config.floatX)
		self.zmean = T.mean(z_sample, axis=0, dtype=theano.config.floatX)
		self.zdiff = T.sum(T.abs_(z_sample[1:] - z_sample[:-1]), axis=0, dtype=theano.config.floatX)
		self.zdiff_mean = T.mean(T.abs_(z_sample[1:] - z_sample[:-1]), axis=0, dtype=theano.config.floatX)
		self.mult_zdiff = T.mean((1-z_sample[:-1]) * z_sample[1:]) + T.mean((1-z_sample[:-1]) * z_sample[1:])
		self.gini_impurity = T.mean(z_sample*(1-z_sample))

		self.z_occlusion = T.mean(z_sample, dtype=theano.config.floatX)


		# z_sample_int = T.cast(z_sample,'int8')
		# test = theano.tensor.and_(z_sample_int[1:],z_sample_int[:-1])

		#This is an alternative to the group lasso (zdiff) which measures the probability that xt = 0 given that xt-1 = 1. Punishing this probability
		#should be another way to encourage coherent rationales
		# if args.hard_attention:

		# else:
		# 	#Sort of a

		#When z is only 0s and 1s, this works out to the aforementioned probability. When it is continuous, this works out to a continuous analog. Sort of.

		#additive version
		ztm1 = z_sample[:-1]
		zt = z_sample[1:]
		self.additive_p0g1 = T.sum((ztm1 - zt)*ztm1)/T.sum(ztm1)

		#multiplicative version
		teq1_tm1eq1 = T.sum((z_sample[:-1]*z_sample[1:]), axis=0, dtype=theano.config.floatX)
		teq0_tm1eq1 =  T.sum((1-z_sample[:-1]*z_sample[1:]), axis=0, dtype=theano.config.floatX)
		self.p0g1 = teq0_tm1eq1/(teq1_tm1eq1+teq0_tm1eq1+0.0001)


		l2_loss = None
		for p in params:
			if l2_loss is None:
				l2_loss = T.sum(p ** 2)
			else:
				l2_loss = l2_loss + T.sum(p ** 2)
		l2_loss = l2_loss * args.l2_reg
		l2_loss.name = 'generator_l2_loss'
		self.l2_loss = l2_loss

		for name, value in self.__dict__.items():
			if hasattr(value,'name'):
				value.name = name


class Encoder(object):
	def __init__(self, args, embedding_layer, nclasses, generator):
		self.args = args
		assert (isinstance(self.args, options.Arguments))
		self.embedding_layer = embedding_layer
		self.nclasses = nclasses
		self.generator = generator

	def ready(self):
		generator = self.generator
		assert(isinstance(generator, Generator))
		embedding_layer = self.embedding_layer
		args = self.args

		padding_id = embedding_layer.vocab_map["<padding>"]

		dropout = generator.dropout

		# len*batch
		x = generator.x.dimshuffle((0, 1, "x"))

		# x = su.sprint(x, 'x at Encoder.ready()')
		# batch*nclasses
		y = self.y = T.fmatrix()

		depth = args.depth
		n_d = args.hidden_dimension

		layers, output_layer, params = self.create_encoder_layers(args, embedding_layer, depth, n_d)
		self.params = params
		self.output_layer = output_layer

		nparams = sum(len(p.get_value(borrow=True).ravel()) for p in params)
		paramsizes = [(p.name, len(p.get_value(borrow=True).ravel())) for p in params]
		iprint("Encoder params: total {}".format(nparams))
		# iprint('\n'.join(['\t{}: {}'.format(name, num) for name, num in paramsizes]))

		padding_masks = T.cast(T.neq(x, padding_id), theano.config.floatX)


		#Create architecture for predicting y, and evaluation of that architecture
		z = self.encoder_z = generator.z_sample.dimshuffle((0, 1, "x"))

		if args.confusion_method == 'shuffle':
			iprint('Using shuffled rationales to confuse primary encoder')
			confused_z = generator.shuffled_z_sample.dimshuffle((0, 1, "x"))
		elif args.confusion_method == 'inverse':
			iprint('Using normal rationales to confuse primary encoder')
			confused_z = generator.z_sample
		elif args.confusion_method == 'flip':
			iprint('Using flipped rationales to confuse primary encoder')
			confused_z = generator.flipped_z_sample.dimshuffle((0, 1, "x"))
		else:
			raise Exception('Unknown encoder confusion method: {}'.format(args.confusion_method))

		py, encoder_h_final, encoder_h_final_size = self.py, self.encoder_h_final, self.encoder_h_final_size = self.pass_x_and_z_through_layers(x, generator.word_embs, z, args, layers, output_layer, padding_masks, depth, n_d, dropout,embedding_layer, return_h_final=True, return_size=True)

		# self.x_z = T.cast(x*z, 'int32')
		# # self.x_z_1hot  = theano.tensor.extra_ops.to_one_hot(self.x_z.dimshuffle((1,0,2)), embedding_layer.n_V)
		# def create_one_hot(bx_z_i):
		# 	bx_z_i_1hot = theano.tensor.extra_ops.to_one_hot(bx_z_i, embedding_layer.n_V).sum(axis=0)
		# 	return bx_z_i_1hot
		#
		# self.x_z_1hot = theano.scan(
		# 	fn = create_one_hot,
		# 	sequences = [self.x_z.dimshuffle((1,0))],
		# )[0]

		confused_py = self.confused_py = self.pass_x_and_z_through_layers(x, generator.word_embs, confused_z, args, layers, output_layer, padding_masks, depth, n_d, dropout,embedding_layer)

		if args.output_distribution:
			# prediction_loss_function = lambda fpy, fy: T.nnet.categorical_crossentropy(fpy, fy).dimshuffle((0,'x'))
			# prediction_loss_function = lambda fpy, fy: -T.sum(fy*T.log(fpy/fy),axis=1).dimshuffle((0,'x'))

			#https://en.wikipedia.org/wiki/Normal_distribution#Other_properties
			iprint('Interpreting prediction output as mean and variance of a gaussian, and calculating KL divergence with input as prediction loss function')

			#KL divergence. Problem with this is that it isn't bounded on the high end.
			prediction_loss_function = lambda fpy,fy: 0.01*((fpy[:,0]-fy[:,0])**2/(2*fy[:,1]**2) + 0.5*(fpy[:,1]**2/fy[:,1]**2-1-T.log(fpy[:,1]**2/fy[:,1]**2))).dimshuffle((0,'x'))

			#Bhattacharyya distance
			# prediction_loss_function = lambda fpy, fy: (0.25*T.log(0.25*(2+fpy[:,1]**2/fy[:,1]**2+py[:,1]**2/fpy[:,1]**2))+0.25*((fpy[:,0]-fy[:,0])**2/(fy[:,1]**2+fpy[:,1]**2))).dimshuffle((0,'x'))

		else:
			prediction_loss_function = lambda fpy, fy: (fpy - fy) ** 2
			iprint('The prediction function outputs a point estimate, so using squared loss as the prediction loss function')

		encoder_prediction_loss_matrix = self.encoder_prediction_loss_matrix = prediction_loss_function(py, y)

		generator_prediction_loss_matrix = self.generator_prediction_loss_matrix = encoder_prediction_loss_matrix

		confused_encoder_prediction_loss_matrix = self.confused_encoder_prediction_loss_matrix = prediction_loss_function(confused_py, y)


		encoder_prediction_loss_vector = self.encoder_prediction_loss_vector= T.mean(encoder_prediction_loss_matrix, axis=1)
		confused_encoder_prediction_loss_vector = self.confused_encoder_prediction_loss_vector= T.mean(confused_encoder_prediction_loss_matrix, axis=1)

		self.weighted_encoder_prediction_loss_vector = args.prediction_loss_weight * encoder_prediction_loss_vector
		encoder_prediction_loss = self.encoder_prediction_loss = T.mean(encoder_prediction_loss_vector)

		generator_prediction_loss_vector = self.generator_prediction_loss_vector = T.mean(generator_prediction_loss_matrix, axis=1)
		self.generator_prediction_loss = T.mean(generator_prediction_loss_vector)

		#Construct architecture for doing inverse prediction, and evaluation of that architecture

		if args.split_encoder: #Create a secondary encoder if necessary
			iprint('Splitting encoder into a primary encoder and a secondary encoder')
			secondary_layers, self.secondary_output_layer, self.secondary_params = self.create_encoder_layers(args, embedding_layer, depth, n_d, prefix='inverse')
			secondary_nparams = sum(len(p.get_value(borrow=True).ravel()) for p in self.secondary_params)
			iprint("Secondary encoder params: total {}".format(secondary_nparams))
			inverse_layers, inverse_output_layer = secondary_layers, self.secondary_output_layer
		else:
			iprint('Using primary encoder to do inverse prediction.')
			inverse_layers, inverse_output_layer = layers, output_layer

		inverse_z = generator.inverse_z_sample.dimshuffle((0, 1, "x"))

		if args.confusion_method == 'shuffle':
			iprint('Using shuffled rationales to confuse secondary encoder')
			confused_inverse_z = generator.shuffled_inverse_z_sample.dimshuffle((0, 1, "x"))
		elif args.confusion_method == 'inverse':
			iprint('Using normal rationales to confuse secondary encoder')
			confused_inverse_z = generator.inverse_z_sample.dimshuffle((0, 1, "x"))
		elif args.confusion_method == 'flip':
			iprint('Using flipped rationales to confuse secondary encoder')
			confused_inverse_z = generator.flipped_inverse_z_sample.dimshuffle((0, 1, "x"))
		else:
			raise Exception('Unknown secondary encoder confusion method: {}'.format(args.confusion_method))



		inverse_py, inverse_h_final = self.inverse_py, inverse_h_final = self.pass_x_and_z_through_layers(x, generator.word_embs, inverse_z, args, inverse_layers, inverse_output_layer, padding_masks, depth, n_d, dropout, embedding_layer, return_h_final=True)
		self.mean_inverse_py = T.mean(inverse_py)
		inverse_encoder_prediction_loss_matrix = self.inverse_encoder_prediction_loss_matrix = prediction_loss_function(inverse_py, y)
		inverse_encoder_prediction_loss_vector = self.inverse_encoder_prediction_loss_vector= T.mean(inverse_encoder_prediction_loss_matrix, axis=1)
		inverse_encoder_prediction_loss = self.inverse_encoder_prediction_loss = T.mean(inverse_encoder_prediction_loss_vector)
		

		confused_inverse_py, _ = self.confused_inverse_py, _ = self.pass_x_and_z_through_layers(x, generator.word_embs, confused_inverse_z, args, inverse_layers, inverse_output_layer, padding_masks, depth, n_d, dropout, embedding_layer, return_h_final=True)
		confused_inverse_encoder_prediction_loss_matrix = self.confused_inverse_encoder_prediction_loss_matrix = prediction_loss_function(confused_inverse_py, y)
		confused_inverse_encoder_prediction_loss_vector = self.confused_inverse_encoder_prediction_loss_vector= T.mean(confused_inverse_encoder_prediction_loss_matrix, axis=1)
		confused_inverse_encoder_prediction_loss = self.confused_inverse_encoder_prediction_loss= T.mean(confused_inverse_encoder_prediction_loss_vector)
		
		
		no_z_inverse_py, _ = self.no_z_inverse_py, _ = self.pass_x_and_z_through_layers(x, generator.word_embs, None, args, inverse_layers, inverse_output_layer, padding_masks, depth, n_d, dropout, embedding_layer, return_h_final=True)
		no_z_inverse_encoder_prediction_loss_matrix = self.no_z_inverse_encoder_prediction_loss_matrix = prediction_loss_function(no_z_inverse_py, y)
		no_z_inverse_encoder_prediction_loss_vector = self.no_z_inverse_encoder_prediction_loss_vector= T.mean(no_z_inverse_encoder_prediction_loss_matrix, axis=1)
		no_z_inverse_encoder_prediction_loss = self.no_z_inverse_encoder_prediction_loss= T.mean(no_z_inverse_encoder_prediction_loss_vector)




		#Figure out what the default prediction would be. This is important for evaluating the inverse encoder.
		zero_z = generator.zero_z.dimshuffle((0, 1, "x"))
		zero_py = self.zero_py = inverse_output_layer.forward(T.zeros_like(encoder_h_final))
		mean_zero_py = self.mean_zero_py = T.mean(zero_py)


		# Meanwhile the generator wants the encoder to do poorly on the inverted rationale, for some definition of poorly
		if args.inverse_generator_loss_type == 'zero':
			iprint('Generator wants inverse predictions to equal zero-prediction of model')
			inverse_generator_prediction_loss_matrix = prediction_loss_function(inverse_py, zero_py)
		elif args.inverse_generator_loss_type == 'diff':
			iprint('Generator wants inverse predictions to be different from rationale predictions')
			inverse_generator_prediction_loss_matrix = 1 - prediction_loss_function(inverse_py, py)
		elif args.inverse_generator_loss_type == 'error':
			iprint('Generator wants inverse predictions to have poor accuracy')
			inverse_generator_prediction_loss_matrix = 1 - prediction_loss_function(inverse_py, y)
		elif args.inverse_generator_loss_type == 'same':
			iprint('Generator wants inverse predictions to have good accuracy')
			inverse_generator_prediction_loss_matrix = prediction_loss_function(inverse_py, y)
		else:
			raise Exception('Unknown inverse generator loss type')

		self.inverse_generator_prediction_loss_matrix = inverse_generator_prediction_loss_matrix
		self.weighted_inverse_generator_prediction_loss_matrix = args.inverse_generator_prediction_loss_weight * inverse_generator_prediction_loss_matrix
		self.weighted_inverse_generator_prediction_loss = T.mean(self.weighted_inverse_generator_prediction_loss_matrix)
		inverse_generator_prediction_loss_vector = self.inverse_generator_prediction_loss_vector =T.mean(inverse_generator_prediction_loss_matrix, axis=1)
		inverse_generator_prediction_loss = self.inverse_generator_prediction_loss = T.mean(inverse_generator_prediction_loss_vector)

		#Construct evaluation of rationale sparsity and coherence
		if args.sparsity_method == 'l1_sum':
			sparsity_loss_vector = generator.zsum
		elif args.sparsity_method == 'l1_norm':
			sparsity_loss_vector = generator.zmean
		else:
			raise Exception('Unknown value for sparsity_method argument: {}'.format(args.sparsity_method))
		self.sparsity_loss_vector = sparsity_loss_vector
		self.weighted_sparsity_loss_vector = args.z_sparsity_loss_weight*sparsity_loss_vector
		self.weighted_sparsity_loss = T.mean(self.weighted_sparsity_loss_vector)


		if args.coherence_method == 'zdiff_sum':
			iprint('Using group lasso (zdiff) to encourage coherence')
			coherence_loss_vector = generator.zdiff
		elif args.coherence_method == 'zdiff_mean':
			iprint('Using mean group lasso (zdiff_mean) to encourage coherence')
			coherence_loss_vector = generator.zdiff_mean
		elif args.coherence_method == 'additive_p0g1':
			iprint('Using additive P(xt=0|xt-1=1) to encourage coherence')
			coherence_loss_vector = generator.additive_p0g1
		elif args.coherence_method == 'p0g1':
			iprint('Using multiplicative P(xt=0|xt-1=1) to encourage coherence')
			coherence_loss_vector = generator.p0g1
		elif args.coherence_method == 'mult_zdiff':
			iprint('Using multiplicative zdiff to encourage coherence')
			coherence_loss_vector = generator.mult_zdiff
		else:
			raise Exception('Unknown value for coherence_method argument: {}'.format(args.coherence_method))
		self.coherence_loss_vector = coherence_loss_vector
		self.weighted_coherence_loss_vector = args.coherence_loss_weight*args.z_sparsity_loss_weight*coherence_loss_vector
		self.weighted_coherence_loss = T.mean(self.weighted_coherence_loss_vector)


		sparsity_loss = self.rationale_sparsity_loss = T.mean(sparsity_loss_vector, axis=0)  * args.z_sparsity_loss_weight
		self.rationale_sparsity_loss.name = 'rationale_sparsity_loss'
		self.rationale_coherence_loss = T.mean(coherence_loss_vector) * args.coherence_loss_weight * args.z_sparsity_loss_weight
		self.rationale_coherence_loss.name = 'rationale_coherence_loss'
		self.gini_impurity_loss = T.mean(generator.gini_impurity)

		#Make a prediction with no z at all. These functions are used when we want to "pretrain" the encoder with no rationale masking.
		no_z_py, no_z_h_final = self.no_z_py, no_z_h_final = self.pass_x_and_z_through_layers(x, generator.word_embs, None, args, layers, output_layer, padding_masks, depth, n_d, dropout, embedding_layer, return_h_final=True)
		no_z_prediction_loss_matrix = self.no_z_prediction_loss_matrix = prediction_loss_function(no_z_py,y)
		no_z_prediction_loss_vector = T.mean(no_z_prediction_loss_matrix, axis=1)
		no_z_prediction_loss = self.no_z_prediction_loss = T.mean(no_z_prediction_loss_vector)


		#Put all pieces together into an overall encoder and generator loss vector (and secondary encoder, if necessary)
		iprint('Constructing generator loss vector from prediction loss')
		iinc()

		# generator_loss_vector = prediction_loss_vector + sparsity_loss_vector * args.z_sparsity_loss_weight + coherence_loss_vector * (args.coherence_loss_weight * args.z_sparsity_loss_weight)

		iprint('Adding prediction loss term with weight {}'.format(args.prediction_loss_weight))
		generator_loss_vector = generator_prediction_loss_vector * args.prediction_loss_weight


		if (args.inverse_generator_prediction_loss_weight != 0):
			iprint('Adding inverse generator prediction loss term with weight {}'.format(args.inverse_generator_prediction_loss_weight))
			generator_loss_vector += inverse_generator_prediction_loss_vector * args.inverse_generator_prediction_loss_weight

		if (args.z_sparsity_loss_weight != 0):
			iprint('Adding rationale sparsity loss term with weight {}'.format(args.z_sparsity_loss_weight))
			generator_loss_vector += sparsity_loss_vector * args.z_sparsity_loss_weight

		if (args.coherence_loss_weight != 0 and args.z_sparsity_loss_weight != 0):
			iprint('Adding rationale coherence loss term with weight {}'.format(args.coherence_loss_weight))
			generator_loss_vector += coherence_loss_vector * (args.coherence_loss_weight * args.z_sparsity_loss_weight)

		idec()
		# generator_loss_vec = loss_vec  + sparsity_loss_vec * args.z_sparsity_loss_weight + coherence_loss_vec * args.coherence_loss_weight


		self.generator_loss_vector = generator_loss_vector
		self.generator_loss = T.mean(generator_loss_vector)


		if args.hard_attention:
			iprint('Using approximate gradient for optimization of generator (hard attention)')
			logpz = generator.logpz
			logpz_vector = self.logpz_sum = T.sum(logpz, axis=0)
			generator_loss_vector_logpz = self.generator_loss_vector_logpz = generator_loss_vector * logpz_vector
			generator_loss_logpz = self.generator_loss_logpz = T.mean(generator_loss_vector_logpz)

			# self.generator_loss_logpz_grad = T.grad(theano.gradient.disconnected_grad(generator_loss_logpz),self.generator.params)

			# loss_g = self.loss_g = theano.gradient.disconnected_grad(generator_loss_logpz * 10  + generator.l2_loss) #todo fix this
			loss_g = self.loss_g = generator_loss_logpz * 10  + generator.l2_loss
			# loss_g = su.sprint(loss_g, 'hard attention loss_g')
		else:
			iprint('Using real gradient for optimization of generator (soft attention)')
			loss_g =  self.loss_g = self.generator_loss * 10 + generator.l2_loss
			# loss_g = su.sprint(loss_g, 'soft attention loss_g')


		if args.generator_architecture.lower() == 'rnn':
			iprint('Using chosen learning rate of {} for generator'.format(args.learning_rate))
			generator_learning_rate = args.learning_rate
		elif args.generator_architecture.lower() == 'sigmoid':
			iprint('Generator is a simple sigmoid applied to each token, so using {} times chosen learning rate {}: {}'.format(args.sigmoid_lr_multiplier, args.learning_rate, args.sigmoid_lr_multiplier*args.learning_rate))
			generator_learning_rate = args.sigmoid_lr_multiplier*args.learning_rate

		self.generator_optimization_updates = create_optimization_updates(
			cost=self.loss_g,
			params=generator.params,
			method=args.learning,
			beta1=args.beta1,
			beta2=args.beta2,
			lr=args.learning_rate,
		)
		self.updates_g, self.lr_g, self.gnorm_g = self.generator_optimization_updates[:3]

		# l2_loss = None
		# for p in params:
		# 	if l2_loss is None:
		# 		l2_loss = T.sum(p ** 2)
		# 	else:
		# 		l2_loss = l2_loss + T.sum(p ** 2)
		l2_loss = self.l2_loss = self.compute_l2_loss(params)* args.l2_reg

		iprint('Constructing encoder loss vector starting with prediction loss')

		if args.use_primary_confusion:
			iprint('Using confusion for primary encoder prediction loss')
			encoder_loss_vector = self.encoder_loss_vector = confused_encoder_prediction_loss_vector
		else:
			iprint('Not using confusion for primary encoder prediction loss')
			encoder_loss_vector = self.encoder_loss_vector = encoder_prediction_loss_vector

		if args.encoder_architecture.lower() == 'rnn':
			iprint('Setting RNN encoder learning rate to chosen rate of {}'.format(args.learning_rate))
			encoder_learning_rate = args.learning_rate
		elif args.encoder_architecture.lower() == 'sigmoid':
			iprint('Encoder is a simple sigmoid, so setting learning rate to {} times chosen rate of {}: {}'.format(args.sigmoid_lr_multiplier, args.learning_rate, args.sigmoid_lr_multiplier*args.learning_rate))
			encoder_learning_rate = args.sigmoid_lr_multiplier*args.learning_rate


		if not args.split_encoder:
			iprint('Not splitting encoder, so adding inverse prediction loss term to primary encoder loss')
			if args.inverse_encoder_prediction_loss_weight > 0:
				iprint('Adding inverse prediction loss to primary encode loss with a weight of {}'.format(args.inverse_encoder_prediction_loss_weight))



				if args.use_confusion:
					iprint('Using confusion for primary encoder inverse prediction loss')
					encoder_loss_vector += confused_inverse_encoder_prediction_loss_vector * args.inverse_encoder_prediction_loss_weight
				else:
					iprint('Not using confusion for primary encoder inverse prediction loss')
					encoder_loss_vector += inverse_encoder_prediction_loss_vector * args.inverse_encoder_prediction_loss_weight

		else:

			# inverse_l2_loss = 0
			# for p in self.secondary_params:
			# 	inverse_l2_loss += T.sum(p ** 2)
			inverse_l2_loss = self.inverse_l2_loss = self.compute_l2_loss(self.secondary_params)* args.l2_reg


			iprint('Adding inverse prediction loss to secondary encoder')

			if args.use_z_for_inverse_encoder_prediction:
				iprint('Using z in the training of the inverse encoder.')
				if args.use_confusion:
					iprint('Using confusion for secondary encoder loss')
					self.inverse_encoder_loss_vector = confused_inverse_encoder_prediction_loss_vector
				else:
					iprint('Not using confusion for secondary encoder loss')
					self.inverse_encoder_loss_vector = inverse_encoder_prediction_loss_vector
			else:
				iprint('Not using z in the training of the inverse encoder (so no point in using confusion)')
				self.inverse_encoder_loss_vector = no_z_inverse_encoder_prediction_loss_vector


			inverse_encoder_loss = T.mean(self.inverse_encoder_loss_vector )
			loss_inverse_e = self.loss_inverse_e = inverse_encoder_loss *10  + inverse_l2_loss
			self.inverse_encoder_optimization_updates = create_optimization_updates(
				cost=loss_inverse_e,
				params=self.secondary_params if not args.fix_bias_term else pu.remove(self.secondary_output_layer.b, self.secondary_params),
				method=args.learning,
				beta1=args.beta1,
				beta2=args.beta2,
				lr=encoder_learning_rate
			)

			self.updates_inverse_e, self.lr_inverse_e, self.gnorm_inverse_e = self.inverse_encoder_optimization_updates[:3]

		encoder_loss = self.encoder_loss = T.mean(encoder_loss_vector)

		self.loss_e =  encoder_loss * 10 + l2_loss


		self.encoder_optimization_updates = create_optimization_updates(
			cost=self.loss_e,
			params=self.params if not args.fix_bias_term else pu.remove(self.output_layer.b, self.params),
			method=args.learning,
			beta1=args.beta1,
			beta2=args.beta2,
			lr=encoder_learning_rate
		)
		self.updates_e, self.lr_e, self.gnorm_e = self.encoder_optimization_updates[:3]


		self.no_z_loss_e = no_z_prediction_loss  + l2_loss
		self.encoder_pretraining_optimization_updates = create_optimization_updates(
			cost=self.no_z_loss_e,
			params=self.params if not args.fix_pretraining_bias_term else pu.remove(self.output_layer.b, self.params),
			method=args.learning,
			beta1=args.beta1,
			beta2=args.beta2,
			lr=encoder_learning_rate
		)
		self.updates_no_z_e, self.no_z_lr_e, self.no_z_gnorm_e = self.encoder_pretraining_optimization_updates[:3]


		#Add a prediction layer that JUST looks at z, not at x*z
		iprint('Creating a z-only encoder')
		z_only_layers, z_only_output_layer, z_only_params = self.z_only_layers, self.z_only_output_layer, self.z_only_params = self.create_encoder_layers(args, 1, depth, n_d, prefix='z_only')

		z_only_py, _, _ = self.z_only_py, _, _ = self.pass_x_and_z_through_layers(None, None, inverse_z, args, z_only_layers, z_only_output_layer, padding_masks, depth, n_d, dropout,embedding_layer, return_h_final=True, return_size=True)
		z_only_encoder_prediction_loss_matrix = self.z_only_encoder_prediction_loss_matrix = prediction_loss_function(z_only_py, y)
		z_only_encoder_prediction_loss_vector = T.mean(z_only_encoder_prediction_loss_matrix, axis=1)
		z_only_encoder_prediction_loss = self.z_only_encoder_prediction_loss = T.mean(z_only_encoder_prediction_loss_vector)

		confused_z_only_py, _, _ = self.confused_z_only_py, _, _ = self.pass_x_and_z_through_layers(None, None, confused_inverse_z, args, z_only_layers, z_only_output_layer, padding_masks, depth, n_d, dropout, embedding_layer, return_h_final=True, return_size=True)
		confused_z_only_encoder_prediction_loss_matrix = self.confused_z_only_encoder_prediction_loss_matrix = prediction_loss_function(z_only_py, y)
		confused_z_only_encoder_prediction_loss_vector = T.mean(confused_z_only_encoder_prediction_loss_matrix, axis=1)
		confused_z_only_encoder_prediction_loss = self.confused_z_only_encoder_prediction_loss = T.mean(confused_z_only_encoder_prediction_loss_vector)

		if args.use_confusion:
			iprint('Using confusion for z-only encoder')
			self.z_only_encoder_loss_vector = confused_z_only_encoder_prediction_loss_vector
		else:
			iprint('Not using confusion for z-only encoder')
			self.z_only_encoder_loss_vector = z_only_encoder_prediction_loss_vector

		self.z_only_encoder_loss = T.mean(self.z_only_encoder_loss_vector )
		self.z_only_l2_loss = self.compute_l2_loss(z_only_params) * args.l2_reg
		self.z_only_loss_e = 10*self.z_only_encoder_loss +self.z_only_l2_loss

		self.z_only_encoder_optimization_updates = create_optimization_updates(
			cost=self.z_only_loss_e,
			params=z_only_params if not args.fix_pretraining_bias_term else pu.remove(z_only_output_layer.b, z_only_params),
			method=args.learning,
			beta1=args.beta1,
			beta2=args.beta2,
			lr=encoder_learning_rate
		)
		self.updates_z_only_e, self.z_only_lr_e, self.z_only_gnorm_e = self.z_only_encoder_optimization_updates[:3]


		self.dumb_z_only_py = T.mean(z, axis=0)
		self.dumb_z_only_prediction_loss = T.mean(prediction_loss_function(self.dumb_z_only_py, y))

		#Name each theano variable appropriately
		for name, value in self.__dict__.items():
			if hasattr(value,'name'):
				value.name = name


	def compute_l2_loss(self,params):
		l2_loss = 0
		for p in params:
			l2_loss += T.sum(p ** 2)
		return l2_loss

	def create_encoder_layers(self, args, embedding_layer, depth, n_d, prefix=''):
		# Set up recurrent layers and output layer
		# n_e = embedding_layer.n_d
		# assert(isinstance(embedding_layer, EmbeddingLayer))
		params = []
		layers = []

		iprint('Creating encoder layers')
		if args.encoder_architecture.lower() == 'rnn':
			iprint('Using an RNN with {} layers and {} activation for encoder'.format(args.layer, args.activation))
			activation = get_activation_by_name(args.activation)
			layer_type = args.layer.lower()

			input_size = embedding_layer.n_d if isinstance(embedding_layer, EmbeddingLayer) else embedding_layer

			for i in xrange(depth):
				if layer_type == "rcnn":
					l = ExtRCNN(
						n_in=input_size if i == 0 else n_d,
						n_out=n_d,
						activation=activation,
						order=args.order,
						name='{}encoder_layer_{}'.format(prefix,i)
					)
				elif layer_type == "lstm":
					l = ExtLSTM(
						n_in=input_size if i == 0 else n_d,
						n_out=n_d,
						activation=activation,
						name='{}encoder_layer_{}'.format(prefix, i)
					)
				layers.append(l)

			if args.use_all:
				size = depth * n_d
			# batch * size (i.e. n_d*depth)
			else:
				size = n_d

			output_layer = Layer(
				n_in=size,
				n_out=self.nclasses,
				activation=sigmoid,
				# initial_bias=args.output_layer_bias,
				name='{}encoder_output_layer'.format(prefix)
			)


		elif args.encoder_architecture.lower() == 'sigmoid':
			iprint('Using a single simple sigmoid layer for the encoder.')
			input_size = embedding_layer.n_d if isinstance(embedding_layer, EmbeddingLayer) else embedding_layer

			output_layer = Layer(
				n_in=input_size,
				n_out=self.nclasses,
				activation=sigmoid,
				# initial_bias=args.output_layer_bias,
				name='{}encoder_output_layer'.format(prefix)
			)

		for l in layers + [output_layer]:
			for p in l.params:
				params.append(p)

		return layers, output_layer, params


	def pass_x_and_z_through_layers(self, x, embs, z, args, layers, output_layer, padding_masks, depth, n_d, dropout, embedding_layer, return_h_final=False, return_size=False):
		'''
		Little utility function that passes an x and a z through some layers to produce a predicted y
		'''

		if args.encoder_architecture.lower() == 'rnn':
			if z is not None:
				masks = padding_masks * z
			else:
				masks = padding_masks

			cnt_non_padding = T.sum(masks, axis=0) + 1e-8
			lst_states = []
			if embs is not None:
				h_prev = embs
			else:
				h_prev = z
			for l in layers:
				h_next = l.forward_all(h_prev, z)
				if args.pooling:  # pooling is off by default
					# batch * n_d
					masked_sum = T.sum(h_next * masks, axis=0)
					lst_states.append(masked_sum / cnt_non_padding)  # mean pooling
				else:
					lst_states.append(h_next[-1])  # last state
				h_prev = apply_dropout(h_next, dropout)

			if args.use_all:
				size = depth * n_d
				# batch * size (i.e. n_d*depth)
				h_final = T.concatenate(lst_states, axis=1)
			else:
				size = n_d
				h_final = lst_states[-1]

			h_final = apply_dropout(h_final, dropout)
			py = output_layer.forward(h_final)

			# # If we are outputting a distribution over possible ys, be sure to normalize
			# iprint('Normalizing output distribution of output layer')
			# if args.output_distribution:
			# 	#If the output distribution is a binary class probability, then normalize it so that it sums to 1
			# 	# if args.output_distribution_interpretation == 'class_probability':
			# 	inverse_sums = (1/T.sum(py,axis=1,keepdims=True))
			# 	py = py * inverse_sums
			# 	# elif args.output_distribution_interpretation == 'one_hot': #otherwise, if it is a one-hot vector, pass the output through a softmax layer
			# 	# 	py = T.softmax(py)


		elif args.encoder_architecture.lower() == 'sigmoid':

			onehot_size = embedding_layer.n_V
			if x is not None and z is not None:
				x_z = T.cast(embs*z, 'int32')
			elif z is not None:
				x_z = T.cast(z, 'int32')
				onehot_size = 2
			elif x is not None:
				x_z = T.cast(embs, 'int32')

			assert(isinstance(embedding_layer,EmbeddingLayer))
			#x_z should be batch_width * batch length (rows correspond to tokens)


			# # self.x_z_1hot  = theano.tensor.extra_ops.to_one_hot(self.x_z.dimshuffle((1,0,2)), embedding_layer.n_V)
			# def create_one_hot(bx_z_i):
			# 	bx_z_i_1hot = theano.tensor.extra_ops.to_one_hot(bx_z_i, onehot_size).sum(axis=0)
			# 	return bx_z_i_1hot
			#
			# x_z_1hot =  theano.scan(
			# 	fn=create_one_hot,
			# 	sequences=[x_z.dimshuffle((1, 0))],
			# )[0]

			if z is not None:
				x_z_centroid = x_z.sum(axis=0)/(z.sum(axis=0) + 0.001)
			else:
				x_z_centroid = x_z.mean(axis=0)

			# x_z_1hot = h_final = theano.tensor.extra_ops.to_one_hot(x_z.T, embedding_layer.n_V)


			py = output_layer.forward(x_z_centroid)
			h_final = x_z_centroid
			size = embedding_layer.n_d

		return_vals = [py]

		if return_h_final:
			return_vals.append(h_final)

		if return_size:
			return_vals.append(size)

		if len(return_vals) == 1:
			return_vals = return_vals[0]

		return return_vals

class Model(object):
	def __init__(self, args, embedding_layer, nclasses, epoch=0, useful_epoch =0, encoder_epochs=0):
		self.args = args
		self.embedding_layer = embedding_layer
		self.padding_id = embedding_layer.vocab_map["<padding>"]
		self.nclasses = nclasses
		self.epochs = epoch  # Number of epochs this model has been trained for
		self.useful_epochs = useful_epoch #Number of epochs that have made useful changes to the model
		self.encoder_epochs=encoder_epochs #Number of epochs the encoder has run on its own prior to training with the generator
		self.is_ready = False

	def ready(self):
		args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
		self.generator = Generator(args, embedding_layer, nclasses)
		self.encoder = Encoder(args, embedding_layer, nclasses, self.generator)
		self.generator.ready()
		self.encoder.ready()
		self.dropout = self.generator.dropout
		self.x = self.generator.x
		self.y = self.encoder.y
		self.z = self.generator.z_sample

		if args.split_encoder: #todo for optimization purposes, it may be nice to include "and args.inverse_generator_prediction_loss_weight > 0", but for understanding model behavior it is useful to have a sense of what the secondary encoder is doing even if it has no impact on the behavior of the generator
			self.params = self.encoder.params + self.generator.params + self.encoder.secondary_params
		else:
			self.params = self.encoder.params + self.generator.params

		# self.explanation_tree = None #A space-preserving tree of explanatory vectors of training and validation_set

		self.compile_functions()
		self.is_ready = True

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
					 self.encoder_epochs
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
					 self.encoder_epochs
					 ),
					fout,
					protocol=pickle.HIGHEST_PROTOCOL
				)

	def load_model(self, path, load_args= False):
		if not os.path.exists(path):
			if path.endswith(".pkl"):
				path += ".gz"
			else:
				path += ".pkl.gz"

		# if self.args.split_encoder:
		with gzip.open(path, "rb") as fin:
			eparams, inverse_eparams, gparams, nclasses, args, epoch, useful_epoch, encoder_epochs = pickle.load(fin)
		# else:
		# 	with gzip.open(path, "rb") as fin:
		# 		eparams, gparams, nclasses, args, epoch, useful_epoch, encoder_epochs = pickle.load(fin)

		# construct model/network using saved configuration
		# self.args = args



		self.nclasses = nclasses
		self.epochs = epoch
		self.useful_epochs = useful_epoch
		self.encoder_epochs = encoder_epochs
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

	def save_result(self, result, result_list, set):
		'''
		Take the dictionary output of one of the theano functions, add a bunch of context to it, and save it into the results list
		:param result:
		:param result_list:
		:param label:
		:return:
		'''
		result['pretraining_epoch'] = self.encoder_epochs
		result['epoch'] = self.epochs
		result['set'] = set
		result['paramset_name'] = self.args.paramset_name
		result['short_paramset_name'] = self.args.short_paramset_name
		result['script_name'] = self.args.script_name
		result['date'] = pu.today()
		result['time'] = pu.now()

		result_list.append(result)

	def load_pretrained_encoder(self,eparams,encoder_epochs):
		for x, v in zip(self.encoder.params, eparams):
			x.set_value(v.get_value())

		if self.args.split_encoder:
			for x, v in zip(self.encoder.secondary_params, eparams):
				x.set_value(v.get_value())

		self.encoder_epochs = encoder_epochs

	# def get_params(self):
	# 	return [x.get_value() for x in self.encoder.params], [x.get_value() for x in self.generator.params], self.nclasses, self.args, self.epochs, self.useful_epochs


	def pretrain(self, train, dev, test):
		return self.train(train, dev, test, encoder_only = True)

	def train(self, train, dev, test, encoder_only = False, dev_rationales = None):


		training_start_time = pu.now()
		iprint('Starting model training at {}'.format(training_start_time))

		args = self.args
		assert(isinstance(args, options.Arguments))
		dropout = self.dropout
		dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)
		dropout.set_value(dropout_prob)


		# if args.use_confusion:
		# 	iprint('Setting confusion probability for secondary encoder to 0.5')
		# 	self.generator.confusion_prob.set_value(0.5)
		# else:
		# 	iprint('Not doing anything to confuse secondary encoder')
		# 	self.generator.confusion_prob.set_value(0.0)


		train_x, train_y = train
		dev_x, dev_y = dev

		if args.output_distribution:
			iprint('Converting input ys to distributions with the following interpretation: {}'.format(args.output_distribution_interpretation))
			train_y, mean_train_y = pu.convert_to_distribution(train_y, args.output_distribution_interpretation)
			dev_y, _ = pu.convert_to_distribution(dev_y,args.output_distribution_interpretation)
		else:
			mean_train_y = np.asarray([np.mean(train_y)])

		baseline_mse = np.mean((train_y-mean_train_y)**2)
		binarized_y = (train_y >= 0.5).astype(float)
		binarized_mean_y = (mean_train_y >= 0.5).astype(float)*np.ones_like(binarized_y)
		baseline_acc = mt.accuracy_score(binarized_y, binarized_mean_y)


		iprint('Mean y in training set is {:.3f}; baseline MSE: {:.3f}; baseline acc: {:.3f}'.format(float(mean_train_y), baseline_mse, baseline_acc))

		padding_id = self.embedding_layer.vocab_map["<padding>"]

		# mean_y = train_y.mean()

		result_list = [] #We'll save output from training and evaluation in here, and then return the whole thing as a df


		#Fix the bias term of the model to produce some desired output in the case of an all-0 rationale.
		if args.bias_term_fix_value is not None and self.epochs == 0:
			if args.bias_term_fix_value == 'mean':
				# If the final output sigmoid layer gets 0 as input from prior layers, then fixing bias term at this value should produce an output that is the mean y value
				bias_term_value = -np.log(1 / mean_train_y - 1, dtype=theano.config.floatX)
				iprint('Mean target value in training set: {}'.format(mean_train_y))
				iprint('Fixing value of encoder bias term to output mean y value with zero input: {}'.format(bias_term_value))
				self.encoder.output_layer.b.set_value(bias_term_value)
				self.encoder.z_only_output_layer.b.set_value(bias_term_value)

				if args.split_encoder:
					self.encoder.secondary_output_layer.b.set_value(bias_term_value)

			elif type(args.bias_term_fix_value) == float:
				desired_output_value = args.bias_term_fix_value
				if not (desired_output_value >= 0.01 and desired_output_value <= 0.99):
					desired_output_value = min(max(desired_output_value, 0.01),0.99)
					iprint('Zero input y set to {}'.format(desired_output_value))
				bias_term_value =  -np.log(1 / desired_output_value - 1, dtype=theano.config.floatX)
				iprint('Bias term to be set so that zero input y  = {}: {}'.format(desired_output_value, bias_term_value))

				self.encoder.output_layer.b.set_value(bias_term_value * np.ones_like(self.encoder.output_layer.b.get_value()))
				self.encoder.z_only_output_layer.b.set_value(bias_term_value * np.ones_like(self.encoder.z_only_output_layer.b.get_value()))

				if args.split_encoder:
					self.encoder.secondary_output_layer.b.set_value(bias_term_value * np.ones_like(self.encoder.output_layer.b.get_value()))
		elif self.epochs > 0:
			iprint('Model has already been trained for {} epochs, so not setting bias term.'.format(self.epochs))
		else:
			iprint('Not setting value of bias term')


		if args.generator_bias_term_fix_value and self.epochs == 0:
			iprint('Setting generator output layer bias term so that default z probability = {}'.format(args.generator_bias_term_fix_value))
			gb_val = pu.invert(args.generator_bias_term_fix_value, 'sigmoid', theano.config.floatX)

			if not args.hard_attention:
				bias_var = self.generator.output_layer.b
			else:
				bias_var = self.generator.output_layer.bias

			iprint('Generator bias term being set to {}'.format(gb_val))
			bias_var.set_value(gb_val * np.ones_like(bias_var.get_value()))

		if dev is not None:
			dev_batches_x, dev_batches_y, dev_batches_i = myio.create_batches(dev_x, dev_y, args.batch, padding_id, return_indices=True)


		start_time = time.time()

		# tick()
		#Create training batches
		if args.split_encoder and args.confusion_method == 'flip':
			iprint('Sorting each training batch by target value to do confusion flip')
			sort_each_batch = True
		else:
			sort_each_batch = False

		train_batches_x, train_batches_y = myio.create_batches(train_x, train_y, args.batch, padding_id, num_policy_samples=args.num_policy_samples, sort_each_batch=sort_each_batch)

		num_batches = len(train_batches_x)

		#Tile training batches if necessary
		if args.num_policy_samples > 1:
			iprint('Tiling each training batch {} times to decrease variance of gradient estimation'.format(args.num_policy_samples))
			for i in range(len(train_batches_x)):
				train_batches_x[i] = np.tile(train_batches_x[i], (1, args.num_policy_samples))
				train_batches_y[i] = np.tile(train_batches_y[i], (args.num_policy_samples, 1))

		# tock('Training batches made')

		# final_dev_pretraining_result = self.run_function_over_batches(dev_batches_x, dev_batches_y, self.evaluate_pretrained_encoder, dropout_prob)
		# # save_result(last_dev_pretraining_result, result_list, 'dev')
		# iprint("Final pretrained model ({} epochs) {}".format(self.encoder_epochs, final_dev_pretraining_result.batch_metric_string()))

		all_update_dict = {str(k.name):v for k,v in self.encoder.updates_g.items()}
		all_update_dict.update({str(k.name):v for k,v in self.encoder.updates_e.items()})
		update_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.y],
			outputs=all_update_dict,
			updates=self.generator.sample_updates
		))




		#We will either only be training the encoder, or
		if encoder_only:
			iprint('Only training encoder (pretraining)')
			epochs = self.encoder_epochs
			max_epochs = args.max_pretrain_epochs
		else:
			iprint('Training both generator and encoder')
			epochs = self.epochs
			max_epochs = args.max_epochs

		useful_epochs = self.useful_epochs

		if epochs < max_epochs :

			iprint('Model has been trained for {} epochs already, max epochs is {}, so training for {} epochs.'.format(self.epochs, args.max_epochs, args.max_epochs - self.epochs))

			if not args.joint_training:
				iprint('Alternating between encoder and generator epochs')

			else:
				iprint('Training encoder and generator together')

			# lr_g = self.generator_optimization_updates[1]
			updates_g, lr_g, gradient_norm_g, gsums_g, xsums_g, max_norm_g = self.encoder.generator_optimization_updates
			# lr_e = self.encoder_optimization_updates[1]
			updates_e, lr_e, gradient_norm_e, gsums_e, xsums_e, max_norm_e = self.encoder.encoder_optimization_updates



			iprint("Training model for {} epochs (maximum)".format(args.max_epochs))
			if args.save_model:
				iprint('Will be saving model parameters to file: {} every epoch'.format(args.save_model))

			eval_period = args.eval_period
			unchanged = 0
			best_dev_generator_loss = 1e+2
			best_dev_e = 1e+2
			last_train_avg_loss = None
			last_dev_avg_loss = None
			# tolerance = 0.005

			max_collapsed = 3
			collapsed = 0 #number of rounds the rationale occlusion has collapsed to near zero or near 1

			if dev:
				# iprint('Initial dev loss')
				# # update_bak = self.generator.sample_updates.items()[0][0].get_value()
				# last_dev_result = self.evaluate_dev_set(dev_batches_x, dev_batches_y, dropout_prob)
				# last_dev_avg_loss = last_dev_result['generator_loss']
				# save_result(last_dev_result, result_list, 'train')
				# self.generator.sample_updates.items()[0][0].set_value(update_bak)
				last_dev_avg_loss = 1e+2


			while epochs < max_epochs and unchanged < args.max_unchanged and collapsed < max_collapsed:
				iprint('Epoch {}...'.format(epochs + 1))

				epoch_start_time = pu.now()
				iinc()

				if args.decay_lr:
					param_bak = [p.get_value(borrow=False) for p in self.params]

				processed = 0


				# for every training batch
				nan_introduced = False
				# tick('Starting training batches')

				encoder_generator_training_results = []
				tick('Training batches')
				for i in xrange(num_batches):
					if (i + 1) % np.ceil(num_batches/5) == 0:
						iprint("{}/{}".format(i + 1, num_batches))

					bx, by = train_batches_x[i], train_batches_y[i]
					mask = bx != padding_id

					# updates_1 = update_function(bx, by)
					# diagnosis_1 = diagnosis_function(bx, by)
					# print '\n'.join([str((k, v.shape, np.sum(v), np.mean(v))) for k, v in diagnosis_1.items()])

					# if args.decay_lr:
					# 	param_bak = [p.get_value(borrow=False) for p in self.params]

					if encoder_only:
						encoder_generator_training_result = self.pretrain_encoder(bx,by)
					else:
						if not args.joint_training:
							encoder_training_result = self.train_encoder(bx, by)
							generator_training_result = self.train_generator(bx, by)
							encoder_generator_training_result = encoder_training_result
							encoder_generator_training_result.update(generator_training_result)
						else:
							if args.split_encoder:
								encoder_generator_training_result =  self.train_both_encoders_and_generator(bx, by)
							else:
								encoder_generator_training_result = self.train_encoder_and_generator(bx, by)
						encoder_generator_training_results.append(encoder_generator_training_result)


					k = len(by)
					processed += k



				tock('Finished training batches.')
				# updates_2 = update_function(bx, by)
				# diagnosis_2 = diagnosis_function(bx, by)
				epochs += 1
				if encoder_only:
					self.encoder_epochs = epochs
				else:
					self.epochs = epochs

				no_improvement = False
				mean_training_result = ModelObjectiveEvaluation(mean_dict_list(encoder_generator_training_results))



				cur_train_avg_loss = mean_training_result['generator_loss']
				self.save_result(mean_training_result, result_list, 'train')


				if 'z_occlusion' in mean_training_result and mean_training_result['z_occlusion'] < 0.01 or mean_training_result['z_occlusion'] > 0.99:
					collapsed += 1
					iprint('Model rationales appear to have collapsed to {}. #{} round this has happened: '.format(int(mean_training_result['z_occlusion']), collapsed))
				else:
					collapsed = 0

				nan_introduced = self.check_if_nan_introduced()


				if nan_introduced:
					pass
				elif not nan_introduced:
					iprint(
					"Training epoch {:.2f} {}".format(
						epochs,
						mean_training_result.batch_metric_string()
					))

					# evaluate performance on dev set
					if dev:
						# tick()
						# cur_dev_result = self.evaluate_dev_set(dev_batches_x, dev_batches_y, dropout_prob)

						self.dropout.set_value(0.0)

						# dev_evaluation_result = self.evaluate_data(dev_batches_x, dev_batches_y, sampling=True)
						batch_evaluation_results = []

						for bx, by in zip(dev_batches_x, dev_batches_y):
							mask = bx != padding_id
							# self.z, self.encoder.generator_loss, self.encoder.prediction_loss, self.encoder.inverse_generator_prediction_loss, self.encoder.rationale_sparsity_loss, self.encoder.pred_diff
							batch_evaluation_result = self.batch_evaluation_function(bx, by)
							batch_evaluation_results.append(batch_evaluation_result)

						# if np.isnan(batch_evaluation_result['py']).any():
						# 	iprint('Nan detected in predicted ys. Pausing.')
						# 	pass

						n = len(dev_batches_x)

						dev_evaluation_result = ModelObjectiveEvaluation(mean_dict_list(batch_evaluation_results))
						if dev_rationales is not None:
							dev_evaluation_result.update(self.calculate_discrete_metrics(batch_evaluation_results, dev_x, dev_y, dev_batches_i, dev_rationales, padding_id))
						iprint("Dev epoch {:.2f}      {}".format(self.epochs,dev_evaluation_result.batch_metric_with_discrete_string()))

						self.dropout.set_value(dropout_prob)

						cur_dev_avg_loss = dev_evaluation_result['generator_loss']
						self.save_result(dev_evaluation_result, result_list, 'dev')

					# tock('Evaluated development set performance')

					# iprint('Encoder param norms:   ' + str(["{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) for x in self.encoder.params]) )
					# iprint('Generator param norms: ' + str(["{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) for x in self.generator.params]))

					# diagnosis_2 = self.diagnosis_function(bx, by)



					# Check to see how we did relative to prior epoch. If worse, recover the previous parameter values. If learning rate decay is on, do that.
						# if the average loss of the training batches is higher than that of the previous batches
						# if cur_train_avg_loss > last_train_avg_loss * (1 + tolerance):
						# 	no_improvement = True
						# 	print("\nNo improvement in train loss {:.4f} --> {:.4f}\n".format(last_train_avg_loss, cur_train_avg_loss))

					# or the same condition holds for the development set
					if dev and cur_dev_avg_loss > last_dev_avg_loss * (1 + args.tolerance):
						no_improvement = True
						iprint("No improvement in dev loss {} --> {}".format(last_dev_avg_loss, cur_dev_avg_loss))

				# diagnosis = self.diagnosis_function(bx, by)

				if no_improvement or nan_introduced:
					# if we did worse on this round on the last one, decrease the learning rate and try again
					# the issue is that I've been getting stuck on this...
					if args.decay_lr:
						if lr_g.get_value() >= args.min_lr:
							lr_val = lr_g.get_value() * 0.5
							lr_val = np.float64(lr_val).astype(theano.config.floatX)
							lr_g.set_value(lr_val)
							lr_e.set_value(lr_val)
							iprint("Decreasing learning rate to {:.5f}".format(float(lr_val)))
						else:
							iprint("Learning rate already at minimum of {:.4f}".format(args.min_lr))
						# Reset the model parameters to their previous (better) configuration

						if args.reset_params:
							iprint('Resetting model parameters')
							for p, v in zip(self.params, param_bak):
								p.set_value(v)
							# tock('Model parameters reset')

							cur_dev_avg_loss = last_dev_avg_loss

					unchanged += 1
				# print 'Unchanged: {}'.format(unchanged)
				else:
					unchanged = 0
					useful_epochs += 1

					if args.save_model and useful_epochs % args.eval_period == 0 :
						iprint('{} useful epochs elapsed (eval period = {}), so saving current model to {}'.format(useful_epochs, args.eval_period, args.save_model))
						self.save_model(args.save_model)
						# tock('Model saved')

				last_train_avg_loss = cur_train_avg_loss
				if dev:
					last_dev_avg_loss = cur_dev_avg_loss

				iprint("Finished epoch at {}, elapsed time {}".format(pu.now(), (pu.now() - epoch_start_time)))

				idec()

			# print(("Training ppoch {:.2f}  prediction loss={:.4f}  generator loss={:.4f}  inverse generator prediction loss={:.4f} rationale sparsity loss={:.4f} p[1]={:.2f}  |g|={:.4f} {:.4f}\t[{:.2f}m / {:.2f}m f/@{}]\n").format(

			# print("\t" + str(["{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) for x in self.encoder.params]) + "\n")
			# print("\t" + str(["{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) for x in self.generator.params]) + "\n")

			if epochs >= max_epochs:
				iprint('Reached maximum number of epochs ({}), so finished training model'.format(args.max_epochs))
			if unchanged >= args.max_unchanged:
				iprint('Reached maximum number of consecutive epochs without model improvement ({}), so finished training model'.format(args.max_unchanged))
			if collapsed >= max_collapsed:
				iprint('Reached maximum number of consecutive epochs with collapsed rationale occlusion ({}), so finished training model'.format(max_collapsed))

			iprint('Saving final version of model')
			self.save_model(args.save_model)
			# self.load_model(args.save_model)

		else:
			iprint('Max epochs {} is less than or equal to current number {}, so not doing any training.'.format(max_epochs, epochs))

		# if dev:
		# 	iprint('Final development set model performance:')
		# 	self.dropout.set_value(0.0)
		#
		#
		# 	# dev_evaluation_result = self.evaluate_data(dev_batches_x, dev_batches_y, sampling=True)
		# 	batch_evaluation_results = []
		#
		# 	for bx, by in zip(dev_batches_x, dev_batches_y):
		# 		mask = bx != padding_id
		# 		# self.z, self.encoder.generator_loss, self.encoder.prediction_loss, self.encoder.inverse_generator_prediction_loss, self.encoder.rationale_sparsity_loss, self.encoder.pred_diff
		# 		batch_evaluation_result = self.batch_evaluation_function(bx, by)
		# 		batch_evaluation_results.append(batch_evaluation_result)
		#
		# 	# if np.isnan(batch_evaluation_result['py']).any():
		# 	# 	iprint('Nan detected in predicted ys. Pausing.')
		# 	# 	pass
		#
		# 	n = len(dev_batches_x)
		#
		# 	dev_evaluation_result = ModelObjectiveEvaluation(mean_dict_list(batch_evaluation_results))
		# 	if dev_rationales is not None:
		# 		dev_evaluation_result.update(self.calculate_discrete_metrics(batch_evaluation_results, dev_x, dev_y, dev_batches_i, dev_rationales, padding_id))
		#
		#
		# 	iprint("Dev epoch {:.2f}      {}".format(self.epochs,dev_evaluation_result.batch_metric_with_discrete_string()))
		#
		# 	self.dropout.set_value(dropout_prob)
		#
		# 	cur_dev_avg_loss = dev_evaluation_result['generator_loss']
		# 	self.save_result(dev_evaluation_result, result_list, 'dev')


		if encoder_only:
			self.encoder_epochs = epochs
		else:
			self.epochs = epochs
		self.useful_epochs = useful_epochs




		iprint("Finished training at {}, elapsed time {}".format(pu.now(), (pu.now() - training_start_time)))


		if args.do_diagnosis:
			self.analyze_architecture(dev_x, dev_y, padding_id, sort_each_batch, args)
		self.dropout.set_value(0)
		self.generator.confusion_prob.set_value(0.0)

		return result_list
	# if args.dev and args.dump:
		# 	print('Dumping rationales for dev data for final version of model\n')
		# 	opath = os.path.join(os.path.dirname(args.dump), 'dev_' + os.path.basename(args.dump))
		# 	self.dump_rationales(opath, dev_batches_x, dev_batches_y,
		# 						 get_loss_and_pred, sample_generator)

	def analyze_architecture(self, dev_x, dev_y, padding_id, sort_each_batch, args):
		iprint('Analyzing architecture by printing values of ALL theano variables that are instances variables in generator or encoder (on a small batch)')
		# A function for inspecting model behavior that outputs every named variable in the model
		output = {k: v for k, v in self.__dict__.items() if type(v) == T.TensorVariable}
		output.update({k: v for k, v in self.generator.__dict__.items() if type(v) == T.TensorVariable})
		output.update({k: v for k, v in self.encoder.__dict__.items() if type(v) == T.TensorVariable})
		diagnosis_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.y],
			outputs=output,
			# givens=[(self.generator.z_sample, self.generator.mle_z)],
			updates=self.generator.sample_updates + self.generator.mle_sample_updates
		))

		small_batches_x, small_batches_y = myio.create_batches(dev_x, dev_y, 10, padding_id, num_policy_samples=args.num_policy_samples, sort_each_batch=sort_each_batch)

		bi = 0
		small_batch_x, small_batch_y = small_batches_x[bi], small_batches_y[bi]
		while (small_batch_x.shape[0] > 20 or not np.any(small_batch_y > 0.8)) and bi < len(small_batches_y) - 1:
			bi += 1
			small_batch_x, small_batch_y = small_batches_x[bi], small_batches_y[bi]

		if bi < len(small_batches_y):
			small_batch_x, small_batch_y = small_batches_x[bi], small_batches_y[bi]

			# confused_diagnosis_3 = diagnosis_function(small_batch_x, small_batch_y )
			diagnosis_3 = diagnosis_function(small_batch_x, small_batch_y)
			diagnosis_3_2 = diagnosis_function(small_batch_x, small_batch_y)

			def print_all(d):
				for k, v in sorted(d.items(), key=lambda (k, v): k):
					iprint('{}:\n{}\n\n'.format(k, v))

			# print '\n'.join([str((k, v.shape, np.sum(v), np.mean(v))) for k, v in diagnosis_3.items()])

			def mask_batch_to_words(x_batch, z_batch, true_y=None, predicted_y=None):
				masked = (x_batch.T * z_batch.T).astype('int')
				underscore_id = int(self.embedding_layer.map_to_ids('_'))
				masked = np.where(masked == 0, underscore_id, masked)

				words = np.asarray([self.embedding_layer.map_to_words(row) for row in x_batch.T])
				masked_words = np.where(z_batch.T == 0, '-', words)
				joined = np.asarray([' '.join(row) for row in masked_words])

				rdf = pd.DataFrame(joined, columns=['masked'])

				if predicted_y is not None:
					rdf['predicted_y'] = predicted_y

				if true_y is not None:
					rdf['true_y'] = true_y

				return rdf

			iprint('\nNo rationale')
			iprint(mask_batch_to_words(small_batch_x, diagnosis_3['zero_z'], true_y=diagnosis_3['y'], predicted_y=diagnosis_3['no_z_py']))

			iprint('\nRationale')
			iprint(mask_batch_to_words(small_batch_x, diagnosis_3['z'], true_y=diagnosis_3['y'], predicted_y=diagnosis_3['py']))

			iprint('\nInverse rationale')
			iprint(mask_batch_to_words(small_batch_x, diagnosis_3['inverse_z_sample'], true_y=diagnosis_3['y'], predicted_y=diagnosis_3['inverse_py']))

			iprint('\nz-only predictor')
			iprint(mask_batch_to_words(small_batch_x, diagnosis_3['z_sample'], true_y=diagnosis_3['y'], predicted_y=diagnosis_3['z_only_py']))
			# pprint(diagnosis_3)

			iprint('\nDumb z-only predictor')
			iprint(mask_batch_to_words(small_batch_x, diagnosis_3['z_sample'], true_y=diagnosis_3['y'], predicted_y=diagnosis_3['dumb_z_only_py']))
			# pprint(diagnosis_3)


			for name, variable in sorted(diagnosis_3.items(), key=lambda t: t[0]):
				iprint('{} {}: '.format(name, variable.shape))
				iinc()
				iprint(variable)
				iprint()
				idec()

		else:
			iprint('Could not find a small enough batch to look at for diagnosis')
		iprint('Done doing analysis')



	def calculate_discrete_metrics(self, evaluation_results, x, y, batches_i, rationales, padding_id, binary_threshold = 0.5):
		'''
		:param evaluation_results: a list of dictionaries, each of which is model output for one batch
		:param y: a list of true y values
		:param batches_i: a batch of index values which indicates how to relate the batches back to the original data
		:param rationales: a vector of true rationales (None if there is no rationale for that item)
		:return:
		'''
		rdict = {
			"y_accuracy": None,
			"y_precision": None,
			"y_recall": None,
			"y_f1": None,

			"rationale_accuracy": None,
			"rationale_precision": None,
			"rationale_recall": None,
			"rationale_f1": None,
		}

		py = [np.nan]*len(y)
		binarized_py = [np.nan]*len(y)
		combined_true_rationale = []
		combined_predicted_rationale = []

		for evaluation_result, batch_i in zip(evaluation_results, batches_i):
			for pyi, i in zip(evaluation_result['py'], batch_i):
				i = int(i)
				py[i] = pyi
				binarized_py[i] = float(pyi >= binary_threshold)

			for pzi, i in zip(evaluation_result['z'].T, batch_i):
				i=int(i)
				if rationales[i] is not None and not np.any(np.isnan(rationales[i])):
					predicted_rationale = np.round(pzi[-len(x[i]):])
					true_rationale = rationales[i]

					assert(len(predicted_rationale) == len(true_rationale))
					combined_true_rationale.extend(true_rationale)
					combined_predicted_rationale.extend(predicted_rationale)

		assert(not np.any(np.isnan(py)))


		binarized_y = (y >= binary_threshold).astype(float)

		rdict['y_accuracy'] = mt.accuracy_score(binarized_y,binarized_py)
		rdict['y_precision'] = mt.precision_score(binarized_y, binarized_py)
		rdict['y_recall'] = mt.recall_score(binarized_y, binarized_py)
		rdict['y_f1'] = mt.f1_score(binarized_y, binarized_py)

		rdict['rationale_accuracy'] = mt.accuracy_score(combined_true_rationale,combined_predicted_rationale)
		rdict['rationale_precision'] = mt.precision_score(combined_true_rationale, combined_predicted_rationale)
		rdict['rationale_recall'] = mt.recall_score(combined_true_rationale, combined_predicted_rationale)
		rdict['rationale_f1'] = mt.f1_score(combined_true_rationale, combined_predicted_rationale)


		return rdict


	def check_if_nan_introduced(self, i = None):
		'''
		Check if there is a NaN in any of the parameters in any of the model components
		:return:
		'''
		nan_introduced = False
		if np.any([np.isnan(param.get_value()).any() for param in self.encoder.params]):
			iprint('Epoch {} batch {} introduced a NaN into the following encoder parameters: {}'.format(self.epochs + 1, i,
																										 [param for param in self.encoder.params if
																										  np.isnan(param.get_value()).any()]))
			nan_introduced = True

		if self.args.split_encoder and np.any([np.isnan(param.get_value()).any() for param in self.encoder.secondary_params]):
			iprint('Epoch {} batch {} introduced a NaN into the following secondary encoder parameters: {}'.format(self.epochs + 1, i,
																												   [param for param in self.encoder.secondary_params
																													if np.isnan(param.get_value()).any()]))
			nan_introduced = True

		if np.any([np.isnan(param.get_value()).any() for param in self.generator.params]):
			iprint('Epoch {} batch {} introduced a NaN into the following generator parameters: {}'.format(self.epochs + 1, i,
																										   [param for param in self.generator.params if
																											np.isnan(param.get_value()).any()]))
			nan_introduced = True
		return nan_introduced

	def run_function_over_batches(self, batches_x, batches_y, func, dropout_prob=None, verbose=False):
		'''
		Convenience function for running a function over a batch of xs and ys and aggregating the results
		:param batches_x:
		:param batches_y:
		:param func:
		:param verbose:
		:return:
		'''
		outputs = []
		iinc()

		if verbose:
			tick('Running function {} on {} batches and aggregating results'.format(func, len(batches_x)))

		# if dropout_prob:
		# 	self.dropout.set_value(0.0)

		for batch_num, (bx, by) in enumerate(zip(batches_x, batches_y)):

			if verbose and batch_num % 100 == 0:
				iprint("{}/{}".format(batch_num, len(batches_x)))
			output = func(bx,by)
			outputs.append(output)

		# if dropout_prob:
		# 	self.dropout.set_value(dropout_prob)
		mean_output = ModelObjectiveEvaluation(mean_dict_list(outputs))

		if verbose:tock('Finished with batches')
		idec()

		return mean_output

	def evaluate_data(self, batches_x, batches_y,  sampling=False):
		padding_id = self.embedding_layer.vocab_map["<padding>"]

		batch_evaluation_results = []

		for bx, by in zip(batches_x, batches_y):

			mask = bx != padding_id
			# self.z, self.encoder.generator_loss, self.encoder.prediction_loss, self.encoder.inverse_generator_prediction_loss, self.encoder.rationale_sparsity_loss, self.encoder.pred_diff
			batch_evaluation_result = self.batch_evaluation_function(bx, by)
			batch_evaluation_results.append(batch_evaluation_result)

			# if np.isnan(batch_evaluation_result['py']).any():
			# 	iprint('Nan detected in predicted ys. Pausing.')
			# 	pass

		n = len(batches_x)

		mean_batch_evaluation_result = ModelObjectiveEvaluation(mean_dict_list(batch_evaluation_results))


		return mean_batch_evaluation_result


	def evaluation_result_to_string(self, dict, order_list = ['encoder_loss', 'generator_loss', 'prediction_loss', 'inverse_generator_prediction_loss', 'inverse_encoder_prediction_loss', 'rationale_sparsity_loss', 'rationale_coherence_loss', 'mean_zero_py', 'z', 'gnorm_e', 'gnorm_g', 'gini_impurity_loss']):
		items = []

		for k,v in sorted(dict.items(), key = lambda (k,v): order_list.index(k) if k in order_list else len(order_list)):
			items.append('{}={:.4f}'.format(k,v))
		return ' '.join(items)

	def evaluate_dev_set(self, dev_batches_x, dev_batches_y,  dropout_prob):
		self.dropout.set_value(0.0)

		dev_evaluation_result = self.evaluate_data(dev_batches_x, dev_batches_y,  sampling=True)

		iprint("Dev epoch {:.2f}      {}".format(
				self.epochs,
				dev_evaluation_result.batch_metric_string()))

		self.dropout.set_value(dropout_prob)


		# cur_dev_avg_loss = dev_evaluation_result['generator_loss']

		return dev_evaluation_result

	class TheanoFunctionWrapper():
		'''
		A wrapper around a theano function that waits to compile it until it is called.
		'''
		def __init__(self, function_generator, wrap_in_evaluation = False):
			self.function_generator = function_generator
			self.function = None
			self.wrap_in_evaluation = wrap_in_evaluation

		def __call__(self, *args, **kwargs):

			if not self.function:
				self.compile()

			if self.wrap_in_evaluation:
				return ModelObjectiveEvaluation(self.function(*args))
			else:
				return self.function(*args)


		def compile(self):
			stack = traceback.extract_stack()
			format_stack = traceback.format_stack()
			index = 0
			for i in range(len(stack) - 1, 0, -1):
				if stack[i][2] == 'compile' and i - 2 > 0:
					index = i - 2
					break
			tick('Compiling theano function for the first time to be used on following line: \n\t{}'.format(format_stack[index]))
			self.function = self.function_generator()
			tock('Done compiling theano function.')



	# #These two functions work together to compile a theano function only the first time it is actually called.
	# def run_theano_function(self, function_generator, wrap_in_evaluation=False):
	# 	func = lambda *args:self.check_function_cache(function_generator,wrap_in_evaluation=wrap_in_evaluation)(*args)
	# 	return func
	#
	# def check_function_cache(self, function_generator, wrap_in_evaluation=False):
	# 	if function_generator.func_code not in self.theano_functions:
	# 		stack = traceback.extract_stack()
	# 		format_stack = traceback.format_stack()
	# 		index = 0
	# 		for i in range(len(stack)-1, 0, -1):
	# 			if stack[i][2] == 'check_function_cache' and i-2 > 0:
	# 				index = i-2
	# 				break
	# 		tick('Compiling theano function for the first time to be used on following line: \n\t{}'.format(format_stack[index]))
	#
	# 		if not wrap_in_evaluation:
	# 			theano_function = function_generator()
	# 			full_function = theano_function
	# 		else:
	# 			theano_function = function_generator()
	# 			full_function = lambda *args:ModelObjectiveEvaluation(theano_function(*args))
	#
	#
	# 		tock('Finished compiling function')
	# 		self.theano_functions[function_generator.func_code] = full_function
	# 	return self.theano_functions[function_generator.func_code]





	def compile_functions(self):
		'''
		Super hacky implementation of just-in-time compilation of all theano functions to save time on precompiling
		:return:
		'''
		args = self.args
		compile_start_time = pu.now()
		iprint('quasi-compiling all theano functions to be used by the model. Start at {}'.format(compile_start_time))
		iinc()
		iprint('...rationale function')
		assert(isinstance(self.generator,Generator))
		self.rationale_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=[self.generator.z_sample, self.generator.inverse_z_sample, self.generator.probs],
			givens=[(self.generator.z_sample, self.generator.mle_z)],
			updates=self.generator.sample_updates
		))


		iprint('...prediction function')
		self.prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.z],
			outputs=[self.encoder.py],
		))

		iprint( '...evaluation function')
		self.evaluation_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.z, self.y],
			outputs={'py':self.encoder.py,
					'inverse_py':self.encoder.inverse_py,
					'zero_py':self.encoder.zero_py,
					'encoder_loss':self.encoder.encoder_loss,
					'generator_loss':self.encoder.generator_loss,
					'prediction_loss':self.encoder.encoder_prediction_loss,
					'inverse_encoder_prediction_loss': self.encoder.inverse_encoder_prediction_loss_matrix,
					'confused_inverse_encoder_prediction_loss': self.encoder.confused_inverse_encoder_prediction_loss_matrix,
					'inverse_generator_prediction_loss':self.encoder.inverse_generator_prediction_loss,
				 	'weighted_inverse_generator_prediction_loss': self.encoder.weighted_inverse_generator_prediction_loss_matrix,
					 'rationale_sparsity_loss':self.encoder.rationale_sparsity_loss,
					 'weighted_rationale_sparsity_loss': self.encoder.weighted_sparsity_loss,

					 'rationale_coherence_loss':self.encoder.rationale_coherence_loss,
					 'weighted_rationale_coherence_loss': self.encoder.weighted_coherence_loss,

					 'no_z_py':self.encoder.no_z_py,
					'no_z_prediction_loss':self.encoder.no_z_prediction_loss},
		), wrap_in_evaluation=True)

		iprint( '...itemwise prediction/evaluation function')
		self.itemwise_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.y],
			outputs={
				'pz':self.generator.z_sample,
				'py':self.encoder.py,
				'inverse_py':self.encoder.inverse_py,
					# 'zero_py':self.encoder.zero_py,
					# 'encoder_loss_vector':self.encoder.encoder_loss_vector,
				'generator_loss':self.encoder.generator_loss_vector,
				'prediction_loss': self.encoder.encoder_prediction_loss_vector,
				'weighted_prediction_loss': self.encoder.weighted_encoder_prediction_loss_vector,
				'sparsity_loss': self.encoder.sparsity_loss_vector,
				'weighted_sparsity_loss': self.encoder.weighted_sparsity_loss_vector,
				'coherence_loss': self.encoder.coherence_loss_vector,
				'weighted_coherence_loss': self.encoder.weighted_coherence_loss_vector,

				'inverse_generator_prediction_loss':self.encoder.inverse_generator_prediction_loss_matrix,
				'weighted_inverse_generator_prediction_loss': self.encoder.weighted_inverse_generator_prediction_loss_matrix,

				'inverse_encoder_prediction_loss': self.encoder.inverse_encoder_loss_vector
			},
			givens=[(self.generator.z_sample, self.generator.mle_z)],
			updates=self.generator.mle_sample_updates
		), wrap_in_evaluation=False)

		iprint( '...itemwise prediction/evaluation function')
		self.itemwise_evaluation_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.z, self.y],
			outputs={'py':self.encoder.py,
					 'inverse_py':self.encoder.inverse_py,
					 'generator_loss': self.encoder.generator_loss_vector,
					 'prediction_loss': self.encoder.encoder_prediction_loss_vector,
					 'weighted_prediction_loss': self.encoder.weighted_encoder_prediction_loss_vector,
					 'sparsity_loss': self.encoder.sparsity_loss_vector,
					 'weighted_sparsity_loss': self.encoder.weighted_sparsity_loss_vector,
					 'coherence_loss': self.encoder.coherence_loss_vector,
					 'weighted_coherence_loss': self.encoder.weighted_coherence_loss_vector,
					 'inverse_generator_prediction_loss': self.encoder.inverse_generator_prediction_loss_matrix,
					 'weighted_inverse_generator_prediction_loss': self.encoder.weighted_inverse_generator_prediction_loss_matrix,
					 'inverse_encoder_prediction_loss': self.encoder.inverse_encoder_loss_vector
					 },

		), wrap_in_evaluation=False)

		# iprint( '...h-final function')
		# self.h_final_function = self.TheanoFunctionWrapper(lambda: theano.function(
		# 	inputs=[self.x, self.z],
		# 	outputs=[self.encoder.w_h_final],
		# ))

		iprint( '...embedding function')
		self.embedding_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=self.generator.word_embs
		))

		# if self.args.retrieval == 'output_weighted_rationale_centroid':
		# 	iprint( '...sequence prediction function')
		# 	self.sequence_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
		# 		inputs=[self.x, self.z],
		# 		outputs=self.encoder.sequence_py
		# 	))

		iprint( '...full z function')
		self.full_z_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=[self.generator.z_sample, self.generator.mle_z, self.generator.probs],
			updates=self.generator.sample_updates
		))



		# iprint("...encoder optimization updates")
		# # create_optimizaion_updates is a generic function, so the calculation of the gradient must be going into encoder.loss_g and encoder.loss_e
		# self.encoder_optimization_updates = create_optimization_updates(
		# 	cost=self.encoder.loss_e,
		# 	params=self.encoder.params if not args.fix_bias_term else pu.remove(self.encoder.output_layer.b, self.encoder.params),
		# 	method=args.learning,
		# 	beta1=args.beta1,
		# 	beta2=args.beta2,
		# 	lr=args.learning_rate
		# )
		# 
		# self.updates_e, lr_e, gnorm_e = self.encoder_optimization_updates[:3]

		# iprint("...generator optimization updates")
		# self.generator_optimization_updates = create_optimization_updates(
		# 	cost=self.encoder.loss_g,
		# 	params=self.generator.params,
		# 	method=args.learning,
		# 	beta1=args.beta1,
		# 	beta2=args.beta2,
		# 	lr=args.learning_rate,
		# )
		# self.updates_g, lr_g, gnorm_g = self.generator_optimization_updates[:3]

		# if args.split_encoder:
		# 	iprint("...inverse encoder optimization updates")
		# 	self.inverse_encoder_optimization_updates = create_optimization_updates(
		# 		cost=self.encoder.loss_inverse_e,
		# 		params=self.encoder.secondary_params if not args.fix_bias_term else pu.remove(self.encoder.secondary_output_layer.b, self.encoder.secondary_params),
		# 		method=args.learning,
		# 		beta1=args.beta1,
		# 		beta2=args.beta2,
		# 		lr=args.learning_rate
		# 	)
		# 
		# 	updates_inverse_e, lr_inverse_e, gnorm_inverse_e = self.inverse_encoder_optimization_updates[:3]

		iprint('Compiling encoder pretraining functions')

		# iprint("...encoder pretraining optimization updates")
		# self.encoder_pretraining_optimization_updates = create_optimization_updates(
		# 	cost=self.encoder.no_z_loss_e,
		# 	params=self.encoder.params if not args.fix_pretraining_bias_term else pu.remove(self.encoder.output_layer.b, self.encoder.params),
		# 	method=args.learning,
		# 	beta1=args.beta1,
		# 	beta2=args.beta2,
		# 	lr=args.learning_rate
		# )
		# 
		# pretrain_updates_e = self.encoder_pretraining_optimization_updates[0]

		iprint('...encoder pretraining function')
		self.pretrain_encoder = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.y],
			outputs={'no_z_prediction_loss':self.encoder.no_z_prediction_loss,
					 'mean_zero_py': self.encoder.mean_zero_py,
					 'l2_loss':self.encoder.l2_loss,
					 'loss_e':self.encoder.no_z_loss_e},
			updates=self.encoder.updates_no_z_e.items()
		), wrap_in_evaluation=True)

		iprint('...encoder pretraining evaluation function')
		self.evaluate_pretrained_encoder = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.y],
			outputs={'no_z_prediction_loss':self.encoder.no_z_prediction_loss,
					 'mean_zero_py': self.encoder.mean_zero_py,
					 'l2_loss':self.encoder.l2_loss,
					 'loss_e':self.encoder.no_z_loss_e},
		), wrap_in_evaluation=True)

		iprint('...no-z prediction evaluation function')
		self.no_z_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=[self.encoder.no_z_py],
		))

		iprint('...full evaluation function')
		self.batch_evaluation_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.y],
			outputs={'z':self.z,
					'encoder_loss':self.encoder.encoder_loss,
					'generator_loss':self.encoder.generator_loss,
					'prediction_loss':self.encoder.encoder_prediction_loss,
					'inverse_generator_prediction_loss':self.encoder.inverse_generator_prediction_loss,
					'inverse_encoder_prediction_loss':self.encoder.inverse_encoder_prediction_loss,
					 'confused_inverse_encoder_prediction_loss': self.encoder.confused_inverse_encoder_prediction_loss,
					 # 'z_only_prediction_loss':self.encoder.z_only_encoder_prediction_loss,
					 # 'dumb_z_only_prediction_loss': self.encoder.dumb_z_only_prediction_loss,
					 'rationale_sparsity_loss':self.encoder.rationale_sparsity_loss,
					'rationale_coherence_loss':self.encoder.rationale_coherence_loss,
					'mean_zero_py':self.encoder.mean_zero_py,
					 'mean_inverse_py': self.encoder.mean_inverse_py,
					 'z_occlusion':self.generator.z_occlusion,
					 'padding_fraction': self.generator.padding_fraction,
					'py':self.encoder.py},
			updates=self.generator.sample_updates
		), wrap_in_evaluation=True)

		if not args.joint_training:
			iprint('Compiling functions for alternating training')

			iprint('...generator training function')

			self.train_generator = self.TheanoFunctionWrapper(lambda: theano.function(
				inputs=[self.x, self.y],
				outputs={'generator_loss':self.encoder.generator_loss,
						 'inverse_generator_prediction_loss':self.encoder.inverse_generator_prediction_loss,
						 'rationale_sparsity_loss':self.encoder.rationale_sparsity_loss,
						 'rationale_coherence_loss':self.encoder.rationale_coherence_loss,
						 'z':self.z,
						 # 'gnorm_g':gnorm_g,
						 'gini_impurity_loss':self.encoder.gini_impurity_loss},
				updates=self.encoder.updates_g.items() + self.generator.sample_updates,
				# mode=theano.compile.MonitorMode(post_func=detect_nan),
				# mode=theano.compile.DebugMode(check_isfinite=True)

			), wrap_in_evaluation=True)

			if args.split_encoder:
				iprint('...encoder training function (with separate updates for inverse encoder)')

				self.train_encoder = self.TheanoFunctionWrapper(lambda: theano.function(
					inputs=[self.x, self.y],
					outputs={'encoder_loss':self.encoder.encoder_loss,
							 'prediction_loss':self.encoder.encoder_prediction_loss,
							 'inverse_encoder_prediction_loss':self.encoder.inverse_encoder_prediction_loss,
							 'mean_zero_py':self.encoder.mean_zero_py,
							 # 'gnorm_e':gnorm_e
							 },

					updates=self.encoder.updates_e.items() + self.encoder.updates_inverse_e.items() + self.generator.sample_updates
				), wrap_in_evaluation=True)

			else:
				iprint('...encoder training function')
				self.train_encoder = self.TheanoFunctionWrapper(lambda: theano.function(
					inputs=[self.x, self.y],
					outputs={'encoder_loss':self.encoder.encoder_loss,
							 'prediction_loss':self.encoder.encoder_prediction_loss,
							 'inverse_encoder_prediction_loss':self.encoder.inverse_encoder_prediction_loss,
							 'mean_zero_py':self.encoder.mean_zero_py,
							 # 'gnorm_e':gnorm_e
							 },

					updates=self.encoder.updates_e.items() + self.generator.sample_updates
				), wrap_in_evaluation=True)

		else:
			iprint('Compiling functions for joint training')

			if args.split_encoder:
				iprint('...joint training function (with separate updates for inverse encoder)')
				combined_updates = self.encoder.updates_e.items() + self.encoder.updates_inverse_e.items()
				if args.fix_rationale_value is None:
					combined_updates += self.encoder.updates_g.items() + self.generator.sample_updates

				self.train_both_encoders_and_generator = self.TheanoFunctionWrapper(lambda: theano.function(
					inputs=[self.x, self.y],
					outputs={'encoder_loss':self.encoder.encoder_loss,
							 'generator_loss':self.encoder.generator_loss,
							 'prediction_loss':self.encoder.encoder_prediction_loss,
							 'inverse_generator_prediction_loss':self.encoder.inverse_generator_prediction_loss,
							 'inverse_encoder_prediction_loss':self.encoder.inverse_encoder_prediction_loss,
							 'confused_inverse_encoder_prediction_loss': self.encoder.confused_inverse_encoder_prediction_loss,
							 # 'z_only_prediction_loss': self.encoder.z_only_encoder_prediction_loss,
							 # 'dumb_z_only_prediction_loss':self.encoder.dumb_z_only_prediction_loss,
							 'rationale_sparsity_loss':self.encoder.rationale_sparsity_loss,
							 'rationale_coherence_loss':self.encoder.rationale_coherence_loss,
							 'mean_zero_py':self.encoder.mean_zero_py,
							 'mean_inverse_py':self.encoder.mean_inverse_py,
							 'z':self.z,
							 'z_occlusion':self.generator.z_occlusion,
							 'padding_fraction':self.generator.padding_fraction,
							 },

					updates=combined_updates,
					# profile=True
					# mode= NanGuardMode(nan_is_error=True, inf_is_error=False, big_is_error=False),

					# mode=theano.compile.MonitorMode(post_func=detect_nan)
					# mode=theano.compile.DebugMode(check_isfinite=True, stability_patience=1, check_c_code=True, check_py_code=False, require_matching_strides=0)

				), wrap_in_evaluation=True)
			else:

				iprint('...joint training function')

				# self.train_encoder_and_generator = self.TheanoFunctionWrapper(lambda: theano.function(
				# 	inputs=[self.x, self.y],
				# 	outputs={'encoder_loss':self.encoder.encoder_loss,
				# 			 'generator_loss':self.encoder.generator_loss,
				# 			 'prediction_loss':self.encoder.prediction_loss,
				# 			 'inverse_generator_prediction_loss':self.encoder.inverse_generator_prediction_loss,
				# 			 'inverse_encoder_prediction_loss':self.encoder.inverse_encoder_prediction_loss,
				# 			 'rationale_sparsity_loss':self.encoder.rationale_sparsity_loss,
				# 			 'rationale_coherence_loss':self.encoder.rationale_coherence_loss,
				# 			 'mean_zero_py':self.encoder.mean_zero_py,
				# 			 'z':self.z,
				# 			 'gnorm_e':gnorm_e,
				# 			 'gnorm_g':gnorm_g,
				# 			 'gini_impurity_loss':self.encoder.gini_impurity_loss},
				#
				# 	updates=self.updates_e.items() + self.updates_g.items() + self.generator.sample_updates,
				# 	# mode=theano.compile.MonitorMode(post_func=detect_nan)
				# 	# mode=theano.compile.DebugMode(check_isfinite=True)
				#
				# ), wrap_in_evaluation=True)
				self.train_encoder_and_generator = self.TheanoFunctionWrapper(lambda: theano.function(
					inputs=[self.x, self.y],
					outputs={
							 'generator_loss': self.encoder.generator_loss,
							 'prediction_loss': self.encoder.encoder_prediction_loss,
							 'rationale_sparsity_loss': self.encoder.rationale_sparsity_loss,
							 'rationale_coherence_loss': self.encoder.rationale_coherence_loss,
							 'z': self.z,
							 # 'gnorm_e': gnorm_e,
							 # 'gnorm_g': gnorm_g
					},

					updates=self.encoder.updates_e.items() + self.encoder.updates_g.items() + self.generator.sample_updates,
					profile=True
					# updates=self.generator.sample_updates,
					# mode=theano.compile.MonitorMode(post_func=detect_nan)
					# mode=theano.compile.DebugMode(check_isfinite=True)

				), wrap_in_evaluation=True)


		# iprint('Normally-compiled prediction function for comparison with original algorithm')
		# prediction_function =         pred_func = theano.function(
		 #        inputs = [ self.x ],
		 #        outputs = [ self.z, self.encoder.py ],
		 #    )


		idec()
		iprint('Finished compiling functions at {}, {} elapsed.'.format(pu.now(),(pu.now() - compile_start_time)))





	def detect_nan(i, node, fn):
		nan_input = np.any([np.isnan(input[0]).any() for input in fn.inputs])
		nan_output = np.any([np.isnan(output[0]).any() for output in fn.outputs])

		# if nan_output and not node.op.__class__ == theano.sandbox.rng_mrg.GPU_mrg_uniform and not node.op.__class__ == theano.sandbox.cuda.basic_ops.GpuAllocEmpty and not node.op.__class__ == theano.sandbox.cuda.basic_ops.GpuIncSubtensor:
		if nan_output and not nan_input:
			iprint('*** NaN detected ***')
			iprint(node.op.__class__)
			iprint('Inputs : %s' % [(input[0].shape, np.isnan(input[0]).any()) for input in fn.inputs])
			iprint('Outputs: %s' % [(output[0].shape, np.isnan(output[0]).any()) for output in fn.outputs])
			iprint(su.trace_apply_node(node))
			iprint()


class ModelObjectiveEvaluation(OrderedDict):
	'''
	Class that holds the result of running or training the model on one or more batches.
	'''

	def __init__(self, in_dict=None):
		super(ModelObjectiveEvaluation, self).__init__()

		# self['py'] = None
		# self['inverse_py'] = None
		# self['zero_py'] = None
		# self['no_z_py'] = None
		# self['mean_zero_py'] = None
		# self['encoder_loss'] = None
		# self['generator_loss'] = None
		# self['prediction_loss'] = None
		# self['inverse_generator_prediction_loss'] = None
		# self['inverse_encoder_prediction_loss'] = None
		# self['no_z_prediction_loss'] = None
		# self['rationale_sparsity_loss'] = None
		# self['rationale_coherence_loss'] = None
		# self['z'] = None
		# self['gnorm_e'] = None
		# self['gnorm_g'] = None
		# self['l2_loss'] = None
		# self['gini_impurity_loss'] = None
		# self['z_occlusion'] = None
		# self['batch_width'] = None
		# self['padding_fraction'] = None

		if in_dict:
			self.update(in_dict)

		#Metrics we want to report for a batch prediction or training run
		self.batch_metrics = ['encoder_loss',
			'generator_loss',
			'prediction_loss',
			'inverse_generator_prediction_loss',
			'inverse_encoder_prediction_loss',
			'weighted_inverse_encoder_prediction_loss',
			'confused_inverse_encoder_prediction_loss',
			'z_only_prediction_loss',
			# 'no_z_prediction_loss',
			'rationale_sparsity_loss',
			'weighted_rationale_sparsity_loss',
			'rationale_coherence_loss',
			'weighted_rationale_coherence_loss',
			'mean_zero_py',
			'z_occlusion',
			# 'batch_width',
			'padding_fraction',
			]

		self.batch_metrics_with_discrete = self.batch_metrics+ [
			'y_f1',
			'rationale_f1',
			'rationale_precision',
			'rationale_recall'
			]

		#Metrics we want to report for a classification of an individual item
		self.item_prediction_metrics = ['py',
			'inverse_py',
			'zero_py',
			'no_z_py',
			'mean_zero_py'] + self.batch_metrics

	def batch_metric_string(self):
		return self.__str__(keys=self.keys(), short=True)

	def batch_metric_with_discrete_string(self):
		return self.__str__(keys=self.keys(), short=True)

	def prediction_metric_string(self, prefix=None, compareto=None):
		return self.__str__(keys=self.item_prediction_metrics, prefix=prefix, compareto=compareto)

	def __str__(self,keys=None, short=False,prefix=None,compareto=None):
		if keys == None:
			keys = self.keys()
		strings = []
		sep = ': ' if not short else '='
		for k in keys:
			if k in self:

				v = self[k]
				if v is not None:
					try:
						string = "{}{}{:.3f}".format(k,sep, float(v))
					except:
						string = "{}{}{}".format(k,sep, v)

					if prefix:
						string = prefix +' '+string


					if compareto:
						try:
							string = string + ' ({:.4f} {})'.format(float(compareto[0][k]),compareto[1])
						except:
							string = string + ' ({} {})'.format(compareto[0][k],compareto[1])
				else:
					# string = '{}{}<not computed>'.format(k,sep)
					pass
				strings.append(string)
		if not short:
			return '\n'.join([s for s in strings])
		else:
			return ' '.join([s for s in strings])