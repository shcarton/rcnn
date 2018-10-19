
import os, sys, gzip
import time
import math
import json
import cPickle as pickle

import numpy as np
os.environ["THEANO_FLAGS"] = "floatX=float32,optimizer=fast_run,on_unused_input='warn',device=gpu3"

import theano
import theano.tensor as T

from nn import create_optimization_updates, get_activation_by_name, sigmoid, linear
from nn import EmbeddingLayer, Layer, LSTM, RCNN, apply_dropout, default_rng
from utils import say
from rationale import myio
import rationale.original_options as options
from rationale.extended_layers import ExtRCNN, ExtLSTM, ZLayer
from collections import OrderedDict
import traceback

class Generator(object):

	def __init__(self, args, embedding_layer, nclasses):
		self.args = args
		self.embedding_layer = embedding_layer
		self.nclasses = nclasses

	def ready(self):
		embedding_layer = self.embedding_layer
		args = self.args
		padding_id = embedding_layer.vocab_map["<padding>"]

		dropout = self.dropout = theano.shared(
				np.float64(args.dropout).astype(theano.config.floatX)
			)

		# len*batch
		x = self.x = T.imatrix()

		n_d = args.hidden_dimension
		n_e = embedding_layer.n_d
		activation = get_activation_by_name(args.activation)

		layers = self.layers = [ ]
		layer_type = args.layer.lower()
		for i in xrange(2):
			if layer_type == "rcnn":
				l = RCNN(
						n_in = n_e,
						n_out = n_d,
						activation = activation,
						order = args.order,
						name = 'generator_layer_{}'.format(i)
					)
			elif layer_type == "lstm":
				l = LSTM(
						n_in = n_e,
						n_out = n_d,
						activation = activation
					)
			layers.append(l)

		# len * batch
		masks = T.cast(T.neq(x, padding_id), theano.config.floatX)

		# (len*batch)*n_e
		embs = embedding_layer.forward(x.ravel())
		# len*batch*n_e
		embs = embs.reshape((x.shape[0], x.shape[1], n_e))
		self.pre_dropout_embs = embs
		embs = apply_dropout(embs, dropout)
		self.word_embs = embs

		flipped_embs = embs[::-1]

		# len*bacth*n_d
		h1 = layers[0].forward_all(embs)
		h2 = layers[1].forward_all(flipped_embs)
		h_final = T.concatenate([h1, h2[::-1]], axis=2)
		self.pre_dropout_h_final = h_final
		h_final = apply_dropout(h_final, dropout)
		self.h_final = h_final
		size = n_d * 2

		if args.layer == 'lstm':
			output_layer = self.output_layer = ZLayer(
				n_in=size,
				n_hidden=args.hidden_dimension2,
				activation=activation,
				rlayer=LSTM(n_in=(size + 1), n_out=args.hidden_dimension2, activation=activation)
			)
		else:
			output_layer = self.output_layer = ZLayer(
				n_in=size,
				n_hidden=args.hidden_dimension2,
				activation=activation,
				rlayer=RCNN((size + 1), args.hidden_dimension2, activation=activation, order=2, name = 'generator_dependent_recurrent_layer'),
				name='generator_dependent_layer'
			)
		# sample z given text (i.e. x)
		z_pred, h, sample_updates = output_layer.sample_all(h_final)
		self.mle_z_pred, _, _ = output_layer.sample_all(h_final,mle=True)

		z_pred = self.z_pred = theano.gradient.disconnected_grad(z_pred)
		self.inverse_z_pred = 1-z_pred


		self.sample_updates = sample_updates
		print "z_pred", z_pred.ndim

		probs = output_layer.forward_all(h_final, z_pred)
		print "probs", probs.ndim

		self.inverse_mle_z_pred = 1-self.mle_z_pred


		logpz = - T.nnet.binary_crossentropy(probs, z_pred) * masks
		logpz = self.logpz = logpz.reshape(x.shape)
		probs = self.probs = probs.reshape(x.shape)

		# batch
		z = z_pred
		self.zsum = T.sum(z, axis=0, dtype=theano.config.floatX)
		self.zdiff = T.sum(T.abs_(z[1:]-z[:-1]), axis=0, dtype=theano.config.floatX)

		params = self.params = [ ]
		for l in layers + [ output_layer ]:
			for p in l.params:
				params.append(p)
		nparams = sum(len(x.get_value(borrow=True).ravel()) \
										for x in params)
		print("total # parameters: {}\n".format(nparams))

		l2_cost = None
		for p in params:
			if l2_cost is None:
				l2_cost = T.sum(p**2)
			else:
				l2_cost = l2_cost + T.sum(p**2)
		l2_cost = l2_cost * args.l2_reg
		self.l2_cost = l2_cost


class Encoder(object):

	def __init__(self, args, embedding_layer, nclasses, generator):
		self.args = args
		self.embedding_layer = embedding_layer
		self.nclasses = nclasses
		self.generator = generator

	def ready(self):
		generator = self.generator
		embedding_layer = self.embedding_layer
		args = self.args
		padding_id = embedding_layer.vocab_map["<padding>"]

		dropout = generator.dropout

		# len*batch
		x = generator.x

		z = generator.z_pred
		z = z.dimshuffle((0,1,"x"))

		# batch*nclasses
		y = self.y = T.fmatrix()

		n_d = args.hidden_dimension
		n_e = embedding_layer.n_d
		activation = get_activation_by_name(args.activation)

		layers = self.layers = [ ]
		depth = args.depth
		layer_type = args.layer.lower()
		for i in xrange(depth):
			if layer_type == "rcnn":
				l = ExtRCNN(
						n_in = n_e if i == 0 else n_d,
						n_out = n_d,
						activation = activation,
						order = args.order,
						name = 'encoder_layer_{}'.format(i)
					)
			elif layer_type == "lstm":
				l = ExtLSTM(
						n_in = n_e if i == 0 else n_d,
						n_out = n_d,
						activation = activation
					)
			layers.append(l)

		# len * batch * 1
		masks = T.cast(T.neq(x, padding_id).dimshuffle((0,1,"x")) * z, theano.config.floatX)
		# batch * 1
		cnt_non_padding = T.sum(masks, axis=0) + 1e-8

		# len*batch*n_e
		embs = generator.word_embs

		pooling = args.pooling
		lst_states = [ ]
		h_prev = embs
		for l in layers:
			# len*batch*n_d
			h_next = l.forward_all(h_prev, z)
			if pooling:
				# batch * n_d
				masked_sum = T.sum(h_next * masks, axis=0)
				lst_states.append(masked_sum/cnt_non_padding) # mean pooling
			else:
				lst_states.append(h_next[-1]) # last state
			h_prev = apply_dropout(h_next, dropout)

		if args.use_all:
			size = depth * n_d
			# batch * size (i.e. n_d*depth)
			h_final = T.concatenate(lst_states, axis=1)
		else:
			size = n_d
			h_final = lst_states[-1]
		h_final = apply_dropout(h_final, dropout)

		output_layer = self.output_layer = Layer(
				n_in = size,
				n_out = self.nclasses,
				activation = sigmoid,
				name = 'encoder_output_layer'
			)

		# batch * nclasses
		preds = self.preds = output_layer.forward(h_final)

		# batch
		loss_mat = self.loss_mat = (preds-y)**2

		pred_diff = self.pred_diff = T.mean(T.max(preds, axis=1) - T.min(preds, axis=1))

		if args.aspect < 0:
			prediction_loss_vector = T.mean(loss_mat, axis=1)
		else:
			assert args.aspect < self.nclasses
			prediction_loss_vector = loss_mat[:,args.aspect]
		self.prediction_loss_vector = prediction_loss_vector

		zsum = generator.zsum
		zdiff = generator.zdiff
		logpz = generator.logpz

		coherent_factor = args.sparsity * args.coherent
		prediction_loss = self.prediction_loss = T.mean(prediction_loss_vector)
		self.prediction_loss.name = 'prediction_loss'
		# sparsity_cost = self.sparsity_cost = T.mean(zsum) * args.sparsity + \
		#                                      T.mean(zdiff) * coherent_factor

		self.sparsity_loss = T.mean(zsum) * args.sparsity
		self.sparsity_loss.name =  'rationale_sparsity_loss'
		self.coherence_loss =  T.mean(zdiff) * coherent_factor
		self.coherence_loss.name = 'rationale_coherence_loss'

		generator_cost_vector = self.generator_cost_vector =  prediction_loss_vector + zsum * args.sparsity + zdiff * coherent_factor
		cost_logpz = T.mean(generator_cost_vector * T.sum(logpz, axis=0))
		self.generator_cost = T.mean(generator_cost_vector)
		self.generator_cost.name = 'generator_cost'

		params = self.params = [ ]
		for l in layers + [ output_layer ]:
			for p in l.params:
				params.append(p)
		nparams = sum(len(x.get_value(borrow=True).ravel()) \
										for x in params)
		print("total # parameters: {}\n".format(nparams))

		l2_cost = None
		for p in params:
			if l2_cost is None:
				l2_cost = T.sum(p**2)
			else:
				l2_cost = l2_cost + T.sum(p**2)
		l2_cost = l2_cost * args.l2_reg
		self.l2_cost = l2_cost

		self.cost_g = cost_logpz * 10 + generator.l2_cost
		self.cost_e = prediction_loss * 10 + l2_cost

class OriginalModel(object):

	def __init__(self, args, embedding_layer, nclasses, epochs = 0):
		print('Model args:')
		print args
		self.args = args
		self.embedding_layer = embedding_layer
		self.nclasses = nclasses
		self.epochs = 0


	def ready(self):
		args, embedding_layer, nclasses = self.args, self.embedding_layer, self.nclasses
		self.generator = Generator(args, embedding_layer, nclasses)
		self.encoder = Encoder(args, embedding_layer, nclasses, self.generator)
		self.generator.ready()
		self.encoder.ready()
		self.dropout = self.generator.dropout
		self.x = self.generator.x
		self.y = self.encoder.y
		self.z = self.generator.z_pred
		self.params = self.encoder.params + self.generator.params
		self.compile_functions()


	def save_model(self, path):
		print('Saving model to {}'.format(path))
		# append file suffix
		if not path.endswith(".pkl.gz"):
			if path.endswith(".pkl"):
				path += ".gz"
			else:
				path += ".pkl.gz"

		# output to path
		with gzip.open(path, "wb") as fout:
			pickle.dump(
				([ x.get_value() for x in self.encoder.params ],   # encoder
				 [ x.get_value() for x in self.generator.params ], # generator
				 self.nclasses,
				 self.args,
				 self.epochs# training configuration
				),
				fout,
				protocol = pickle.HIGHEST_PROTOCOL
			)


	def load_model(self, path):
		print('Loading model from {}'.format(path))
		if not os.path.exists(path):
			if path.endswith(".pkl"):
				path += ".gz"
			else:
				path += ".pkl.gz"

		# try:
		with gzip.open(path, "rb") as fin:
			eparams, gparams, nclasses, args, self.epochs  = pickle.load(fin)
		print('Loaded model has been trained for {} epochs'.format(self.epochs))

		# except:
		#     with gzip.open(path, "rb") as fin2:
		#         eparams, gparams, nclasses, args = pickle.load(fin2)
		#         self.epochs = 0
		#     print('Could not load epoch information from saved model, so setting trained epochs to 0')



		# construct model/network using saved configuration
		# self.args = args
		self.nclasses = nclasses
		self.ready()
		for x,v in zip(self.encoder.params, eparams):
			x.set_value(v)
		for x,v in zip(self.generator.params, gparams):
			x.set_value(v)


	def train(self, train, dev, test, rationale_data=None):
		print('Beginning training for original dependent model')
		args = self.args
		dropout = self.dropout
		padding_id = self.embedding_layer.vocab_map["<padding>"]


		print('{} epochs trained out of a max of {}, so training for {} epochs').format(self.epochs, args.max_epochs, args.max_epochs-self.epochs)

		if self.epochs < args.max_epochs:

			if dev is not None:
				dev_batches_x, dev_batches_y = myio.create_batches(
								dev[0], dev[1], args.batch, padding_id
							)
			if test is not None:
				test_batches_x, test_batches_y = myio.create_batches(
								test[0], test[1], args.batch, padding_id
							)
			if rationale_data is not None:
				valid_batches_x, valid_batches_y = myio.create_batches(
						[ u["xids"] for u in rationale_data ],
						[ u["y"] for u in rationale_data ],
						args.batch,
						padding_id,
						sort = False
					)

			start_time = time.time()
			train_batches_x, train_batches_y = myio.create_batches(
								train[0], train[1], args.batch, padding_id
							)
			print("{:.2f}s to create training batches\n\n".format(
					time.time()-start_time
				))

			print('Creating encoder optimization updates')
			updates_e, lr_e, gradient_norm_e = create_optimization_updates(
								   cost = self.encoder.cost_e,
								   params = self.encoder.params,
								   method = args.learning,
								   beta1 = args.beta1,
								   beta2 = args.beta2,
								   lr = args.learning_rate
							)[:3]

			print('Creating generator optimization updates')
			updates_g, lr_g, gradient_norm_g = create_optimization_updates(
								   cost = self.encoder.cost_g,
								   params = self.generator.params,
								   method = args.learning,
								   beta1 = args.beta1,
								   beta2 = args.beta2,
								   lr = args.learning_rate
							)[:3]

			print('Compiling sample generator function')
			sample_generator = self.TheanoFunctionWrapper(lambda : theano.function(
					inputs = [ self.x ],
					outputs = self.z,
					updates = self.generator.sample_updates
				))

			print('Creating get_loss_and_pred function')
			get_loss_and_pred = self.TheanoFunctionWrapper(lambda : theano.function(
					inputs = [ self.x, self.y ],
					outputs = [self.encoder.prediction_loss_vector, self.encoder.preds, self.z],
					updates = self.generator.sample_updates
				))
			print('Creating eval_generator function')
			eval_generator = self.TheanoFunctionWrapper(lambda : theano.function(
					inputs = [ self.x, self.y ],
					outputs = [self.z, self.encoder.generator_cost, self.encoder.prediction_loss,
							   self.encoder.pred_diff],
					updates = self.generator.sample_updates
				))
			print('Creating train_generator function')
			train_encoder_and_generator = self.TheanoFunctionWrapper(lambda : theano.function(
					inputs = [ self.x, self.y ],
					outputs = [ self.encoder.generator_cost,
								self.encoder.prediction_loss,
								self.encoder.sparsity_loss,
								self.encoder.coherence_loss,
								self.z,
								gradient_norm_e,
								gradient_norm_g ],
					updates = updates_e.items() + updates_g.items() + self.generator.sample_updates,
					# updates = self.generator.sample_updates,
				))

			eval_period = args.eval_period
			unchanged = 0
			best_dev = 1e+2
			best_dev_e = 1e+2
			last_train_avg_cost = None
			last_dev_avg_cost = None
			tolerance = 0.10 + 1e-3
			dropout_prob = np.float64(args.dropout).astype(theano.config.floatX)

			all_update_dict = {str(k.name): v for k, v in updates_g.items()}
			all_update_dict.update({str(k.name): v for k, v in updates_e.items()})
			update_function = self.TheanoFunctionWrapper(lambda: theano.function(
				inputs=[self.x, self.y],
				outputs=all_update_dict,
				updates=self.generator.sample_updates
			))

			diagnosis_function = self.TheanoFunctionWrapper(lambda: theano.function(
				inputs=[self.x, self.y],
				outputs=
				{
					'logpz': self.generator.logpz,
					'probs': self.generator.probs,
					'pz': self.generator.z_pred,
					'py': self.y,
					'cost_g': self.encoder.cost_g,
					'cost_e': self.encoder.cost_e,
					'mle_pz': self.generator.mle_z_pred,
					'prediction_loss_vector': self.encoder.prediction_loss_vector,
					'generator_loss_vector': self.encoder.generator_cost_vector,
					'zdiff': self.generator.zdiff,
					'zsum': self.generator.zsum,
					'h_final': self.generator.h_final,
					'pre_dropout_h_final': self.generator.pre_dropout_h_final,
					'pre_dropout_embs': self.generator.pre_dropout_embs,
					'embs': self.generator.word_embs

				},
				updates=self.generator.sample_updates
			))

			# for epoch in xrange(args.max_epochs):
			while self.epochs < args.max_epochs:
				epoch = self.epochs
				unchanged += 1
				if unchanged > 20: return

				train_batches_x, train_batches_y = myio.create_batches(
								train[0], train[1], args.batch, padding_id
							)

				more = True
				if args.decay_lr:
					param_bak = [ p.get_value(borrow=False) for p in self.params ]

				while more:
					processed = 0
					train_generator_cost = 0.0
					train_prediction_loss = 0.0
					train_sparsity_loss = 0.0
					train_coherence_loss = 0.0
					p1 = 0.0
					start_time = time.time()

					N = len(train_batches_x)

					for i in xrange(N):
						if (i+1) % int(N/5) == 0:
							print("\r{}/{} {:.2f}       ".format(i+1,N,p1/(i+1)))

						bx, by = train_batches_x[i], train_batches_y[i]
						mask = bx != padding_id

						# updates_1 = update_function(bx, by)
						# diagnosis_1 = diagnosis_function(bx, by)

						generator_cost, prediction_loss, sparsity_loss, coherence_loss, bz, gl2_e, gl2_g = train_encoder_and_generator(bx, by)

						k = len(by)
						processed += k
						train_generator_cost += generator_cost
						train_prediction_loss += prediction_loss
						train_sparsity_loss += sparsity_loss
						train_coherence_loss += coherence_loss


						p1 += np.sum(bz*mask) / (np.sum(mask)+1e-8)

					# updates_2 = update_function(bx, by)
					# diagnosis_2 = diagnosis_function(bx, by)

					cur_train_avg_cost = train_generator_cost / N

					if dev:
						self.dropout.set_value(0.0)
						dev_obj, dev_loss, dev_diff, dev_p1 = self.evaluate_data(
								dev_batches_x, dev_batches_y, eval_generator, sampling=True)
						self.dropout.set_value(dropout_prob)
						cur_dev_avg_cost = dev_obj

					more = False
					if args.decay_lr and last_train_avg_cost is not None:
						if cur_train_avg_cost > last_train_avg_cost*(1+tolerance):
							more = True
							print("\nTrain cost {} --> {}\n".format(
									last_train_avg_cost, cur_train_avg_cost
								))
						if dev and cur_dev_avg_cost > last_dev_avg_cost*(1+tolerance):
							more = True
							print("\nDev cost {} --> {}\n".format(
									last_dev_avg_cost, cur_dev_avg_cost
								))

					if more and lr_g.get_value() > 0.000001:
						lr_val = lr_g.get_value()*0.5
						lr_val = np.float64(lr_val).astype(theano.config.floatX)
						lr_g.set_value(lr_val)
						lr_e.set_value(lr_val)
						print("Decrease learning rate to {}\n".format(float(lr_val)))
						for p, v in zip(self.params, param_bak):
							p.set_value(v)
						continue

					last_train_avg_cost = cur_train_avg_cost
					if dev: last_dev_avg_cost = cur_dev_avg_cost

					print("\n")
					print(("Generator Epoch {:.2f}  generator cost={:.4f} prediction loss = {:.4f} sparsity loss={:.4f} coherence loss={:.4f} p[1]={:.2f}  |g|={:.4f} {:.4f}\t[{:.2f}m / {:.2f}m]\n").format(
							epoch+(i+1.0)/N,
							train_generator_cost / N,
						train_prediction_loss / N,
						train_sparsity_loss / N,
							train_coherence_loss / N,
							p1 / N,
							float(gl2_e),
							float(gl2_g),
							(time.time()-start_time)/60.0,
							(time.time()-start_time)/60.0/(i+1)*N
						))
					print("\t"+str([ "{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
									for x in self.encoder.params ])+"\n")
					print("\t"+str([ "{:.2f}".format(np.linalg.norm(x.get_value(borrow=True))) \
									for x in self.generator.params ])+"\n")
					self.epochs += 1

					if dev:
						if dev_obj < best_dev:
							best_dev = dev_obj
							unchanged = 0
							if args.dump and rationale_data:
								self.dump_rationales(args.dump, valid_batches_x, valid_batches_y,
											get_loss_and_pred, sample_generator)

							if args.save_model:
								self.save_model(args.save_model)

						print(("\tsampling devg={:.4f}  mseg={:.4f}  avg_diffg={:.4f}" +
									"  p[1]g={:.2f}  best_dev={:.4f}\n").format(
							dev_obj,
							dev_loss,
							dev_diff,
							dev_p1,
							best_dev
						))

						if rationale_data is not None:
							self.dropout.set_value(0.0)
							r_mse, r_p1, r_prec1, r_prec2 = self.evaluate_rationale(
									rationale_data, valid_batches_x,
									valid_batches_y, eval_generator)
							self.dropout.set_value(dropout_prob)
							print(("\trationale mser={:.4f}  p[1]r={:.2f}  prec1={:.4f}" +
										"  prec2={:.4f}\n").format(
									r_mse,
									r_p1,
									r_prec1,
									r_prec2
							))

					# diagnosis = self.diagnosis_function(bx, by)
					# print('')

	def evaluate_data(self, batches_x, batches_y, eval_func, sampling=False):
		padding_id = self.embedding_layer.vocab_map["<padding>"]
		tot_obj, tot_mse, tot_diff, p1 = 0.0, 0.0, 0.0, 0.0
		for bx, by in zip(batches_x, batches_y):
			if not sampling:
				e, d = eval_func(bx, by)
			else:
				mask = bx != padding_id
				bz, o, e, d = eval_func(bx, by)
				p1 += np.sum(bz*mask) / (np.sum(mask) + 1e-8)
				tot_obj += o
			tot_mse += e
			tot_diff += d
		n = len(batches_x)
		if not sampling:
			return tot_mse/n, tot_diff/n
		return tot_obj/n, tot_mse/n, tot_diff/n, p1/n

	def evaluate_rationale(self, reviews, batches_x, batches_y, eval_func):
		args = self.args
		padding_id = self.embedding_layer.vocab_map["<padding>"]
		aspect = str(args.aspect)
		p1, tot_mse, tot_prec1, tot_prec2 = 0.0, 0.0, 0.0, 0.0
		tot_z, tot_n = 1e-10, 1e-10
		cnt = 0
		for bx, by in zip(batches_x, batches_y):
			mask = bx != padding_id
			bz, o, e, d = eval_func(bx, by)
			tot_mse += e
			p1 += np.sum(bz*mask)/(np.sum(mask) + 1e-8)
			if args.aspect >= 0:
				for z,m in zip(bz.T, mask.T):
					z = [ vz for vz,vm in zip(z,m) if vm ]
					assert len(z) == len(reviews[cnt]["xids"])
					truez_intvals = reviews[cnt][aspect]
					prec = sum( 1 for i, zi in enumerate(z) if zi>0 and \
								any(i>=u[0] and i<u[1] for u in truez_intvals) )
					nz = sum(z)
					if nz > 0:
						tot_prec1 += prec/(nz+0.0)
						tot_n += 1
					tot_prec2 += prec
					tot_z += nz
					cnt += 1
		#assert cnt == len(reviews)
		n = len(batches_x)
		return tot_mse/n, p1/n, tot_prec1/tot_n, tot_prec2/tot_z

	def dump_rationales(self, path, batches_x, batches_y, eval_func, sample_func):
		embedding_layer = self.embedding_layer
		padding_id = self.embedding_layer.vocab_map["<padding>"]
		lst = [ ]
		for bx, by in zip(batches_x, batches_y):
			loss_vec_r, preds_r, bz = eval_func(bx, by)
			assert len(loss_vec_r) == bx.shape[1]
			for loss_r, p_r, x,y,z in zip(loss_vec_r, preds_r, bx.T, by, bz.T):
				loss_r = float(loss_r)
				p_r, x, y, z = p_r.tolist(), x.tolist(), y.tolist(), z.tolist()
				w = embedding_layer.map_to_words(x)
				r = [ u if v == 1 else "__" for u,v in zip(w,z) ]
				diff = max(y)-min(y)
				lst.append((diff, loss_r, r, w, x, y, z, p_r))

		#lst = sorted(lst, key=lambda x: (len(x[3]), x[2]))
		with open(path,"w") as fout:
			for diff, loss_r, r, w, x, y, z, p_r in lst:
				fout.write( json.dumps( { "diff": diff,
										  "loss_r": loss_r,
										  "rationale": " ".join(r),
										  "text": " ".join(w),
										  "x": x,
										  "z": z,
										  "y": y,
										  "p_r": p_r } ) + "\n" )


	def compile_functions(self):
		print('Compiling additional functions needed by original code')

		print('...maximum likelihood z function')
		self.maximum_likelihood_z_function =  self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=[self.generator.mle_z_pred, self.generator.inverse_mle_z_pred, self.generator.z_pred, self.generator.inverse_z_pred, self.generator.probs],
			updates=self.generator.sample_updates,
		))

		print ('...embedding function')
		self.embedding_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x],
			outputs=self.encoder.embs
		))

		print("...evaluation prediction function")
		self.evaluation_prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.z, self.y],
			outputs={'preds': self.encoder.preds,
					 'encoder_loss': self.encoder.prediction_loss,
					 'generator_loss': self.encoder.generator_cost,
					 'prediction_loss': self.encoder.prediction_loss,
					 'rationale_sparsity_cost': self.encoder.sparsity_loss,
					 'rationale_coherence_cost': self.encoder.coherence_loss},
		), wrap_in_evaluation=True)

		print('...prediction function')
		self.prediction_function = self.TheanoFunctionWrapper(lambda: theano.function(
			inputs=[self.x, self.z],
			outputs=[self.encoder.preds],
		))

	class TheanoFunctionWrapper():
		'''
		A wrapper around a theano function that waits to compile it until it is called.
		'''

		def __init__(self, function_generator, wrap_in_evaluation=False):
			self.function_generator = function_generator
			self.function = None
			self.wrap_in_evaluation = wrap_in_evaluation

		def __call__(self, *args, **kwargs):

			if not self.function:
				self.compile()

			if self.wrap_in_evaluation:
				return OriginalModel.ModelObjectiveEvaluation(self.function(*args))
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
			start_time = time.time()
			print('Tick {} | Compiling theano function for the first time to be used on following line: \n\t{}'.format(start_time, format_stack[index]))
			self.function = self.function_generator()
			end_time = time.time()
			print('Tock. {} | {} elapsed. Done compiling theano function.'.format(end_time, end_time-start_time))


	class ModelObjectiveEvaluation(OrderedDict):
		'''
		Class that holds the result of running or training the model on one or more batches.
		'''

		def __init__(self, in_dict=None):
			super(OriginalModel.ModelObjectiveEvaluation, self).__init__()

			self['preds'] = np.NaN
			self['inverse_preds'] = np.NaN
			self['zero_preds'] = np.NaN
			self['no_z_preds'] = np.NaN
			self['mean_zero_preds'] = np.NaN
			self['encoder_loss'] = np.NaN
			self['generator_loss'] = np.NaN
			self['prediction_loss'] = np.NaN
			self['inverse_generator_prediction_loss'] = np.NaN
			self['inverse_encoder_prediction_loss'] = np.NaN
			self['no_z_prediction_loss'] = np.NaN
			self['rationale_sparsity_cost'] = np.NaN
			self['rationale_coherence_cost'] = np.NaN
			self['z'] = np.NaN
			self['gnorm_e'] = np.NaN
			self['gnorm_g'] = np.NaN
			self['l2_cost'] = np.NaN
			self['gini_impurity_cost'] = np.NaN

			if in_dict:
				self.update(in_dict)

			# Metrics we want to report for a batch prediction or training run
			self.batch_metrics = ['encoder_loss',
								  'generator_loss',
								  'prediction_loss',
								  'inverse_generator_prediction_loss',
								  'inverse_encoder_prediction_loss',
								  'no_z_prediction_loss',
								  'rationale_sparsity_cost',
								  'rationale_coherence_cost',
								  'gnorm_e',
								  'gnorm_g',
								  'gini_impurity_cost',
								  'mean_zero_preds',
								  'l2_cost']

			# Metrics we want to report for a classification of an individual item
			self.prediction_metrics = ['preds',
									   'inverse_preds',
									   'zero_preds',
									   'no_z_preds',
									   'mean_zero_preds'] + self.batch_metrics

		def batch_metric_string(self):
			return self.__str__(keys=self.batch_metrics, short=True)

		def prediction_metric_string(self, prefix=None, compareto=None):
			return self.__str__(keys=self.prediction_metrics, prefix=prefix, compareto=compareto)

		def __str__(self, keys=None, short=False, prefix=None, compareto=None):
			if keys == None:
				keys = self.keys()
			strings = []
			sep = ': ' if not short else '='
			for k in keys:
				v = self[k]
				if not np.isnan(v):
					try:
						string = "{}{}{:.3f}".format(k, sep, float(v))
					except:
						string = "{}{}{}".format(k, sep, v)

					if prefix:
						string = prefix + ' ' + string

					if compareto:
						try:
							string = string + ' ({:.4f} {})'.format(float(compareto[0][k]), compareto[1])
						except:
							string = string + ' ({} {})'.format(compareto[0][k], compareto[1])

					strings.append(string)
			if not short:
				return '\n'.join([s for s in strings])
			else:
				return ' '.join([s for s in strings])

def main():
	print args
	assert args.embedding, "Pre-trained word embeddings required."

	embedding_layer = myio.create_embedding_layer(
						args.embedding
					)

	max_len = args.max_len

	if args.train:
		train_x, train_y = myio.read_annotations(args.train)
		train_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in train_x ]

	if args.dev:
		dev_x, dev_y = myio.read_annotations(args.dev)
		dev_x = [ embedding_layer.map_to_ids(x)[:max_len] for x in dev_x ]

	if args.load_rationale:
		rationale_data = myio.read_rationales(args.load_rationale)
		for x in rationale_data:
			x["xids"] = embedding_layer.map_to_ids(x["x"])

	if args.train:
		model = OriginalModel(
					args = args,
					embedding_layer = embedding_layer,
					nclasses = len(train_y[0])
				)
		model.ready()

		#debug_func2 = theano.function(
		#        inputs = [ model.x, model.z ],
		#        outputs = model.generator.logpz
		#    )
		#theano.printing.debugprint(debug_func2)
		#return

		model.train(
				(train_x, train_y),
				(dev_x, dev_y) if args.dev else None,
				None, #(test_x, test_y),
				rationale_data if args.load_rationale else None
			)

	if args.load_model and args.dev and not args.train:
		model = OriginalModel(
					args = None,
					embedding_layer = embedding_layer,
					nclasses = -1
				)
		model.load_model(args.load_model)
		print("model loaded successfully.\n")

		# compile an evaluation function
		eval_func = theano.function(
				inputs = [ model.x, model.y ],
				outputs = [model.z, model.encoder.generator_cost, model.encoder.prediction_loss,
						   model.encoder.pred_diff],
				updates = model.generator.sample_updates
			)

		# compile a predictor function
		pred_func = theano.function(
				inputs = [ model.x ],
				outputs = [ model.z, model.encoder.preds ],
				updates = model.generator.sample_updates
			)

		# batching data
		padding_id = embedding_layer.vocab_map["<padding>"]
		dev_batches_x, dev_batches_y = myio.create_batches(
						dev_x, dev_y, args.batch, padding_id
					)

		# disable dropout
		model.dropout.set_value(0.0)
		dev_obj, dev_loss, dev_diff, dev_p1 = model.evaluate_data(
				dev_batches_x, dev_batches_y, eval_func, sampling=True)
		print("{} {} {} {}\n".format(dev_obj, dev_loss, dev_diff, dev_p1))


if __name__=="__main__":
	args = options.load_arguments()
	main()