import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn import create_optimization_updates, get_activation_by_name, sigmoid, linear
from nn import EmbeddingLayer, Layer, RecurrentLayer, LSTM, RCNN, apply_dropout, default_rng
from nn import create_shared, random_init
import sam_util as su

class ExtRCNN(RCNN):

	def __init__(self, mask_input=False, *args, **kwargs):
		super(ExtRCNN, self).__init__(*args,**kwargs)
		self.mask_input=mask_input

	def forward(self, x_t, mask_t, hc_tm1):

		if self.mask_input:
			hc_t = super(ExtRCNN, self).forward(mask_t*x_t, hc_tm1)
		else:
			hc_t = super(ExtRCNN, self).forward(x_t, hc_tm1)
			hc_t = mask_t * hc_t + (1-mask_t) * hc_tm1
		return hc_t

	def forward_all(self, x, mask, h0=None, return_c=False):
		# x is len*batch*n_d
		if h0 is None:
			if x.ndim > 1:
				h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)
			else:
				h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)

		if mask is not None:
			h, _ = theano.scan(
						fn = self.forward,
						sequences = [ x, mask ],
						outputs_info = [ h0 ]
					)
		else:
			h, _ = theano.scan(
				fn=super(ExtRCNN, self).forward,
				sequences=[x],
				outputs_info=[h0]
			)
		if return_c:
			return h
		elif x.ndim > 1:
			return h[:,:,self.n_out*self.order:]
		else:
			return h[:,self.n_out*self.order:]

	def copy_params(self, from_obj):
		self.internal_layers = from_obj.internal_layers
		self.bias = from_obj.bias

class ExtLSTM(LSTM):

	def __init__(self, mask_input=False, *args, **kwargs):
		super(ExtLSTM, self).__init__(*args,**kwargs)
		self.mask_input=mask_input

	def forward(self, x_t, mask_t, hc_tm1):
		if self.mask_input:
			hc_t = super(ExtLSTM, self).forward(mask_t*x_t, hc_tm1)
		else:
			hc_t = super(ExtLSTM, self).forward(x_t, hc_tm1)
			hc_t = mask_t * hc_t + (1-mask_t) * hc_tm1
		return hc_t

	def forward_all(self, x, mask, h0=None, return_c=False):
		if h0 is None:
			if x.ndim > 1:
				# h0 = T.zeros((x.shape[1], self.n_out*(self.order+1)), dtype=theano.config.floatX)

				h0 = T.zeros((x.shape[1], self.n_out * 2), dtype=theano.config.floatX)
			else:
				# h0 = T.zeros((self.n_out*(self.order+1),), dtype=theano.config.floatX)
				h0 = T.zeros((self.n_out * 2,), dtype=theano.config.floatX)

		if mask is not None: #TODO: This really should not be necessary, but I can't figure out how to do this with Python polymorphism
			h, _ = theano.scan(
						fn = self.forward,
						sequences = [ x, mask ],
						outputs_info = [ h0 ]
					)
		else:
			h, _ = theano.scan(
				fn=super(ExtLSTM, self).forward,
				sequences=[x],
				outputs_info=[h0]
			)
		if return_c:
			return h
		elif x.ndim > 1:
			return h[:, :, self.n_out:]
		else:
			return h[:, self.n_out:]



	def copy_params(self, from_obj):
		self.internal_layers = from_obj.internal_layers

class ZLayer(object):
	def __init__(self, n_in, n_hidden, activation, rlayer,name='ZLayer'):
		self.n_in, self.n_hidden, self.activation = \
				n_in, n_hidden, activation
		# self.MRG_rng = MRG_RandomStreams() #TODO change this back if switching doesn't work
		self.MRG_rng = T.shared_randomstreams.RandomStreams()
		self.rlayer = rlayer
		self.layers = [ rlayer ]
		self.name=name

		self.create_parameters()

	def create_parameters(self):
		n_in, n_hidden = self.n_in, self.n_hidden
		activation = self.activation

		self.w1 = create_shared(random_init((n_in,)), name="{}_w1".format(self.name))
		self.w2 = create_shared(random_init((n_hidden,)), name="{}_w2".format(self.name))
		bias_val = random_init((1,))[0]
		self.bias = theano.shared(np.cast[theano.config.floatX](bias_val), name = "{}_bias".format(self.name))

		# rlayer = RCNN((n_in+1), n_hidden, activation=activation, order=2)
		#
		#
		# self.rlayer = rlayer
		# self.layers = [ rlayer ]

	# def forward_with_counter(self, x_t, z_):

	def forward(self, x_t, z_t, h_tm1, pz_tm1):

		print "z_t", z_t.ndim

		pz_t = sigmoid(
					T.dot(x_t, self.w1) +
					T.dot(h_tm1[:,-self.n_hidden:], self.w2) +
					self.bias
				)

		xz_t =  T.concatenate([x_t, z_t.reshape((-1,1))], axis=1)
		h_t = self.rlayer.forward(xz_t, h_tm1)

		# batch
		return h_t, pz_t

	def forward_all(self, x, z):
		assert x.ndim == 3
		assert z.ndim == 2
		xz = T.concatenate([x, z.dimshuffle((0,1,"x"))], axis=2)
		h0 = T.zeros((1, x.shape[1], self.n_hidden), dtype=theano.config.floatX)
		h = self.rlayer.forward_all(xz)
		h_prev = T.concatenate([h0, h[:-1]], axis=0)
		assert h.ndim == 3
		assert h_prev.ndim == 3
		pz = sigmoid(
				T.dot(x, self.w1) +
				T.dot(h_prev, self.w2) +
				self.bias
			)
		assert pz.ndim == 2
		return pz


	def sample(self, x_t, z_tm1, h_tm1, mle=False):
		'''
		Given an input xt and the output and hidden state from the previous timestep,
		this method samples an output z_t for this timestep. It does this by calculating
		the continuous output of the model, and then using that as a probability to sample from
		:param x_t:
		:param z_tm1:
		:param h_tm1:
		:return:
		'''
		# print 'Zlayer sample function'
		# print "z_tm1", z_tm1.ndim, type(z_tm1)
		# print "x_t", x_t.ndim, type(x_t)
		# print "h_tm1", h_tm1.ndim, type(h_tm1)

		pz_t = sigmoid(
					T.dot(x_t, self.w1) +
					T.dot(h_tm1[:,-self.n_hidden:], self.w2) +
					self.bias
				)

		# batch
		pz_t = pz_t.ravel()


		if mle:
			z_t = T.cast(pz_t >= 0.5, theano.config.floatX)
		else:
			z_t = T.cast(self.MRG_rng.binomial(size=pz_t.shape,
										p=pz_t), theano.config.floatX)

		z_t = theano.gradient.disconnected_grad(z_t)

		xz_t = T.concatenate([x_t, z_t.reshape((-1,1))], axis=1)
		h_t = self.rlayer.forward(xz_t, h_tm1)

		return z_t, pz_t, h_t

	def mle_sample(self, *args):
		return self.sample(*args,mle=True)

	def sample_with_count(self, x_t, z_tm1, cz_tm1, h_tm1, mle=False):
		pz_t = sigmoid(
			T.dot(x_t, self.w1) +
			T.dot(h_tm1[:, -self.n_hidden:], self.w2) +
			self.bias
		)

		# batch
		pz_t = pz_t.ravel()

		if mle:
			z_t = T.cast(pz_t >= 0.5, theano.config.floatX)
		else:
			z_t = T.cast(self.MRG_rng.binomial(size=pz_t.shape,
											   p=pz_t), theano.config.floatX)

		z_t = theano.gradient.disconnected_grad(z_t)
		cz_t = cz_tm1 + z_t

		xz_t = T.concatenate([x_t, z_t.reshape((-1, 1)), cz_t.reshape((-1, 1))], axis=1)
		h_t = self.rlayer.forward(xz_t, h_tm1)

		return z_t, pz_t, cz_t, h_t


	def mle_sample_with_count(self, *args):
		return self.sample_with_count(*args,mle=True)

	def sample_all(self, x, mle = False, add_counter = False):
		'''

		:param x:
		:return:
		'''

		#Reset RNG to get more predictable results
		# self.MRG_rng = MRG_RandomStreams()

		if type(self.rlayer) == RCNN:
			h0 = T.zeros((x.shape[1], self.n_hidden*(self.rlayer.order+1)), dtype=theano.config.floatX)
		else:
			h0 = T.zeros((x.shape[1], self.n_hidden*2), dtype=theano.config.floatX)

		z0 = T.zeros((x.shape[1],), dtype=theano.config.floatX)
		# x = su.sprint(x, 'x as seen by ZLayer.sample_all')

		cz0 = T.zeros((x.shape[1],), dtype=theano.config.floatX)

		if add_counter:
			if mle:
				([ z, pz, cz, h ], updates) = theano.scan(
									fn = self.mle_sample_with_count,
									sequences = [ x ],
									outputs_info = [ z0,  None, cz0, h0 ]
				)
			else:
				([ z, pz, cz, h ], updates) = theano.scan(
									fn = self.sample_with_count,
									sequences = [ x ],
									outputs_info = [ z0, None, cz0, h0 ]
						)

		else:

			if mle:
				([ z, pz, h ], updates) = theano.scan(
									fn = self.mle_sample,
									sequences = [ x ],
									outputs_info = [ z0,  None, h0 ]
				)
			else:
				([ z, pz,  h ], updates) = theano.scan(
									fn = self.sample,
									sequences = [ x ],
									outputs_info = [ z0, None, h0 ]
						)
		assert z.ndim == 2

		if add_counter:
			return z,pz,cz,h,updates
		else:
			return z, pz, h, updates



	# def sample_multi(self, x_t, sz_tm1, h_tm1, num_samples=3):
	#	 '''
	#
	#	 :param x_t: batch size * 2n_d
	#	 :param sz_tm1: batch size * num_samples * 1
	#	 :param h_tm1: batch size * num_samples *  2n_d
	#
	#	 :param num_samples:
	#	 :return:
	#	 sz_t: batch size * num_samples * 1
	#	 h_t: batch size * num_samples *  2n_d
	#
	#	 '''
	#	 # print 'Zlayer multi-sample function'
	#	 # print "z_tm1", sz_tm1.ndim, type(sz_tm1)
	#	 # print "x_t", x_t.ndim, type(x_t)
	#	 # print "h_tm1", h_tm1.ndim, type(h_tm1)
	#	 # print "Num samples", num_samples
	#
	#	 sz_tm1 = self.sprint(sz_tm1, 'sz_tm1')
	#	 x_t = self.sprint(x_t, 'x_t')
	#	 h_tm1 = self.sprint(h_tm1, 'h_tm1 (initial)')
	#
	#
	#	 print 'Variable w1 shape:', self.w1.get_value().shape
	#
	#	 t1 = T.shape_padright(T.dot(x_t, self.w1)) #batch size * 0
	#	 # print 'T1 attrs',dir(t1)
	#
	#	 t1 = self.sprint(t1, 't1')
	#
	#	 # self.w2 = self.sprint(self.w2, 'w2')
	#	 print 'Variable w2 shape:', self.w2.get_value().shape
	#	 h_tm1_2 = self.sprint(h_tm1[:, :, -self.n_hidden:],'h_tm1_2')
	#
	#	 t2 =  T.dot(h_tm1_2, self.w2) # batch size * num_samples
	#	 # print 'T2 attrs',dir(t2)
	#
	#	 t2 = self.sprint(t2, 't2')
	#
	#	 print 'Variable bias shape:', self.bias.get_value().shape
	#
	#
	#	 t21 = self.sprint(t2+t1,'t21')
	#	 t21b = self.sprint(t21 + self.bias,'t21b')
	#
	#	 pz_t = self.pz_t= sigmoid(
	#		 t21b
	#	 )
	#	 pz_t = self.sprint(pz_t, 'pz_t')
	#
	#
	#	 # pz_t batch size * num_samples * 1
	#	 pz_t = pz_t.ravel()
	#	 sz_t = T.cast(self.MRG_rng.binomial(size=pz_t.shape,
	#									 p=pz_t), theano.config.floatX)
	#
	#
	#
	#	 #Calculate the probability of each z value that was actually drawn
	#	 # spz_t = (1-sz_t) + 2*(sz_t-0.5)*pz_t
	#
	#	 sz_t = sz_t.reshape((-1,  num_samples, 1)) #batch size * num_samples * 1
	#	 sz_t = self.sprint(sz_t, 'sz_t')
	#
	#	 x_tr = T.tile(x_t.reshape((x_t.shape[0], 1, x_t.shape[1])),[1,num_samples,1]) #batch size * 1 * x _width
	#
	#	 x_tr = self.sprint(x_tr, 'x_tr')
	#
	#
	#	 xz_t = T.concatenate([x_tr, sz_t], axis=2) #batch size * num_samples * x_width+1
	#	 xz_t = self.sprint(xz_t, 'xz_t')
	#
	#	 xz_td = xz_t.dimshuffle(0, 2, 1)
	#	 xz_td = self.sprint(xz_td, 'xz_td')
	#
	#	 h_tm1 = self.sprint(h_tm1, 'h_tm1 (again)')
	#	 h_tm1d = h_tm1.dimshuffle(0,2,1)
	#	 h_tm1d = self.sprint(h_tm1d, 'h_tm1d')
	#
	#
	#	 h_t = self.rlayer.forward(xz_td, h_tm1d) #batch size * num_samples *  2n_d
	#
	#	 h_t = self.sprint(h_t, 'h_t')
	#
	#	 # print "sz_t", sz_t.ndim, type(sz_t)
	#	 # print "xz_t", xz_t.ndim, type(xz_t)
	#	 # print "h_t", h_t.ndim, type(h_t)
	#
	#	 return sz_t, h_t
	#
	# def sample_all_multi(self, x, num_samples=1):
	#	 '''
	#	 Sample multiple rationales for a batch x.
	#
	#	 :param x: x-width * batch size * 2n_d
	#	 :param num_samples:
	#	 :return:
	#	 sz:  batch size * x width * num_samples
	#	 '''
	#	 if type(self.rlayer) == RCNN:
	#		 print 'Prepping zlayer h0 for RCNN'
	#		 h0 = T.zeros((x.shape[1],  num_samples,  self.n_hidden*(self.rlayer.order+1)), dtype=theano.config.floatX)
	#	 else:
	#		 print 'Prepping zlayer h0 for LSTM'
	#		 h0 = T.zeros((x.shape[1], num_samples, self.n_hidden*2), dtype=theano.config.floatX)
	#
	#	 #z0 should be batch-size * num_samples * 1
	#	 sz0 = T.zeros((x.shape[1], num_samples,1), dtype=theano.config.floatX)
	#	 # sz0 = T.zeros((x.shape[1],num_sample), dtype=theano.config.floatX)  ??
	#
	#	 # print 'Zlayer multi-sample-all function'
	#	 # print "x", x.ndim, type(x)
	#	 # print "sz0", sz0.ndim, type(sz0)
	#	 # print "h0", h0.ndim, type(h0)
	#
	#	 ([ sz, h ], updates) = theano.scan(fn = self.sample_multi,sequences = [ x ],outputs_info = [sz0, h0])
	#
	#	 # assert sz.ndim == 3
	#	 return sz, updates


	@property
	def params(self):
		return [ x for layer in self.layers for x in layer.params ] + \
			   [ self.w1, self.w2, self.bias ]

	@params.setter
	def params(self, param_list):
		start = 0
		for layer in self.layers:
			end = start + len(layer.params)
			layer.params = param_list[start:end]
			start = end
		self.w1.set_value(param_list[-3].get_value())
		self.w2.set_value(param_list[-2].get_value())
		self.bias.set_value(param_list[-1].get_value())

