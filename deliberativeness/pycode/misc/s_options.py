import sys

def load_arguments(d = None, args = None):

	if not args:
		args = Arguments()

	if d:
		for k, v in d.items():
			if k in args.__dict__:
				setattr(args, k, v)
			else:
				raise Exception('Argument "{}" does not exist in class Arguments'.format(k))
	else:
		print 'Using default values for all arguments'

	if len(sys.argv) > 2:
		raise Exception('Command-line arguments not supported yet')

	return args


class Arguments():

	def __str__(self):
		return str(self.__dict__)

	def __repr__(self):
		return repr(self.__dict__)

	def __init__(self):
		self.fix_rationale_value = None  # Value to fix rationale at. 1 or 0, or None else for not to do it
		self.load_rationale = ""  # path to annotated rationale data

		self.embedding = ""  # path to pre-trained word vectors

		self.embedding_size = 200  # word embedding size

		self.save_model = ""  # path to save model parameters

		self.load_model = ""  # path to load model

		self.train = ""  # path to training data

		self.dev = ""  # path to development data

		self.test = ""  # path to test data

		self.dump = ""  # path to dump rationale

		self.max_epochs = 100  # maximum # of epochs

		self.eval_period = -1  # evaluate model every k examples

		self.batch = 256  # mini-batch size

		self.learning = "adam"  # learning method

		self.learning_rate = 0.0005  # learning rate

		self.dropout = 0.1  # dropout probability

		self.l2_reg = 1e-6  # L2 regularization weight

		self.activation = "tanh" #type of activation function

		self.d = self.hidden_dimension = 200 #hidden dimension

		self.d2 = self.hidden_dimension2 = 30 #other hidden dimension

		self.layer = "rcnn"  # type of recurrent layer

		self.depth = 1  # number of layers

		self.pooling = 0  # whether to use mean pooling or the last state

		self.order = 2  # feature filter width

		self.use_all = 0  # whether to use the states of all layers

		self.max_len = -1  # max number of words in input

		self.sparsity = 0.0003

		self.coherent = 2.0

		self.aspect = -1

		self.beta1 = 0.9

		self.beta2 = 0.999

		self.decay_lr = False

		self.reset_params = False

		self.min_lr = 1.0e-6

		self.fix_emb = 1

		self.contrastive_z_loss = 0  # If this is on, then instead of regularizing the generator in the normal way, we instead encourage its inverse to be of the opposite class

		self.prediction_loss_weight = 1.0  # How to weight the prediction loss in the cost function

		self.inverse_encoder_prediction_loss_weight = 0  # If contrastive z-loss is on, this is how much this term should be weighted

		self.inverse_generator_prediction_loss_weight = 0  # If contrastive z-loss is on, this is how much this term should be weighted

		self.use_z_for_inverse_encoder_prediction = True # If true, train the inverse encoder with z. If not, just train it on the raw data.

		self.z_sparsity_loss_weight = 0  # If contrastive z-loss is on, this is how much the size of the rationale should be penalized

		self.coherence_loss_weight = 0  # If contrastive z-loss is on, this is how much the size of the rationale should be penalized

		self.gini_impurity_weight = 0  # Punishes gini impurity in z probabilities. More relevant in soft-attention mode where z = z probs

		self.retrieval = 'final_h'  # What retrieval method to use. Currently includes h_final, rationale_centroid, rationale_bigram_centroid and output_weighted_rationale_centroid

		self.output_layer_bias = None  # What value to use as an initial value for the bias term of the output layer for the prediction model. This essentially defines what you want the model to do in the absence of any information (i.e. what to do if the rationale obscures the whole text)

		self.max_unchanged = 10  # How many consecutive epochs a model can experience no improvement before the training process quits

		self.num_policy_samples = 1  # How many times we should sample in estimating the rationale policy gradients

		self.fix_bias_term = None  # Whether or not to fix the bias term at some value.

		self.use_confusion = True  # Whether to try to confuse the secondary encoder during training

		self.use_primary_confusion = False  # Whether to try to confuse the primary encoder during training

		self.bidirectional_generator = True  # Whether the generator should be bidirectional or not

		self.fix_pretraining_bias_term = None  # Whether or not to fix the bias term of the pretrained model at some value

		self.bias_term_fix_value = None  # What value to fix bias term at. Possible values include False, "mean" for mean y value of training set, or a float

		self.generator_bias_term_fix_value = None #How to initialize the bias term for the generator

		self.encoder_pretraining_epochs = 0  # How many epochs to pretrain the encoder for

		self.tolerance = 0.005  # How much improvement an epoch has to have to be considered an improvement

		self.dependent = False  # Whether to use the version of the algorithm where each zi is dependent or independent of the sampled value of zi-1

		self.joint_training = True  # Whether to jointly or alternatively train the encoder and generator

		self.encoder_pretrain_epochs = -1  # How many epochs to pretrain the encoder for, if any

		self.coherence_method = 'zdiff'  # How to encourage the raitonale to be more globby. Either zdiff or p0g1

		self.split_encoder = True  # Whether or not to split the encoder into two models; one which tries to do well on the rationale and one which tries to do well on the antirationale

		self.hard_attention = True  # Whether the rationale whould be implemented as a hard attention mechanism via stochastic semilinear units and a REINFORCE optimization algorith, or as soft attention.

		self.inverse_generator_loss_type = 'zero'  # What metric should be used to punish the generator on the encoder's performance on the inverse rationale. 'error', 'zero', 'diff'

		self.confusion_method = 'flip'  # What additional information should be shown to the secondary encoder to confuse it. Options include: 'shuffle', 'inverse', 'flip'

		self.rng_seed = 123  # Seed to use for secondary encoder mask shuffling

		self.model_mode = 'new'  # Whether to use the original unmodified model code, the new model code, or the scikit-learn code. Acceptable values include 'new', 'original', and 'sklearn', and 'lime'

		self.sklearn_l1_C = 1.0 #parameter to pass into C when using 'sklearn' model mode

		self.subsample_training = -1  # How to subsample the training set

		self.sparsity_method = 'l1_sum'  # How to measure sparsity of the rationale mask

		self.output_distribution = False  # Whether to output a distribution rather than a value

		self.output_distribution_interpretation = 'regression'  # How to convert from input target values to a target distribution. In the 'one_hot' case we construct a one-hot vector with a 1.0 value in the appropriate bucket. In the 'class_probability' case we treat the target value as the probability of the positive class

		self.do_diagnosis = False #Whether to run a "diagnosis" function at the end of the training function to look at the values of all variables

		self.return_c = False #Whether the generator RNN should return its hidden state as well as its output

		self.mask_input = False #Whether to mask the input to the encoder with 0s, or to mask the hidden state

		self.add_counter = False

		self.generator_architecture = 'rnn' #Whether the generator should be an RNN or a single sigmoid (logistic regression) layer. 'rnn' or 'sigmoid'

		self.encoder_architecture = 'rnn' #Whether the encoder should be an RNN or a single sigmoid (logistic regression) layer. 'rnn' or 'sigmoid'

		self.sigmoid_lr_multiplier = 100 #How much to multiply the learning rate by for sigmoid layers

