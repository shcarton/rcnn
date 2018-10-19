import sys
import argparse
from collections import OrderedDict

def load_arguments():
	argparser = argparse.ArgumentParser(sys.argv[0])


	argparser.add_argument("--fix_rationale_value",
						   type=int,
						   default=-1,
						   help="Value to fix rationale at. 1 or 0, or anything else for not to do it"
						   )


	argparser.add_argument("--load_rationale",
						   type=str,
						   default="",
						   help="path to annotated rationale data"
						   )
	argparser.add_argument("--embedding",
						   type=str,
						   default="",
						   help="path to pre-trained word vectors"
						   )
	argparser.add_argument("--embedding_size",
						   type=int,
						   default=200,
						   help="word embedding size"
						   )
	argparser.add_argument("--save_model",
						   type=str,
						   default="",
						   help="path to save model parameters"
						   )
	argparser.add_argument("--load_model",
						   type=str,
						   default="",
						   help="path to load model"
						   )
	argparser.add_argument("--train",
						   type=str,
						   default="",
						   help="path to training data"
						   )
	argparser.add_argument("--dev",
						   type=str,
						   default="",
						   help="path to development data"
						   )
	argparser.add_argument("--test",
						   type=str,
						   default="",
						   help="path to test data"
						   )
	argparser.add_argument("--dump",
						   type=str,
						   default="",
						   help="path to dump rationale"
						   )
	argparser.add_argument("--max_epochs",
						   type=int,
						   default=100,
						   help="maximum # of epochs"
						   )
	argparser.add_argument("--eval_period",
						   type=int,
						   default=-1,
						   help="evaluate model every k examples"
						   )
	argparser.add_argument("--batch",
						   type=int,
						   default=256,
						   help="mini-batch size"
						   )
	argparser.add_argument("--learning",
						   type=str,
						   default="adam",
						   help="learning method"
						   )
	argparser.add_argument("--learning_rate",
						   type=float,
						   default=0.0005,
						   help="learning rate"
						   )
	argparser.add_argument("--dropout",
						   type=float,
						   default=0.1,
						   help="dropout probability"
						   )
	argparser.add_argument("--l2_reg",
						   type=float,
						   default=1e-6,
						   help="L2 regularization weight"
						   )
	argparser.add_argument("-act", "--activation",
						   type=str,
						   default="tanh",
						   help="type of activation function"
						   )
	argparser.add_argument("-d", "--hidden_dimension",
						   type=int,
						   default=200,
						   help="hidden dimension"
						   )
	argparser.add_argument("-d2", "--hidden_dimension2",
						   type=int,
						   default=30,
						   help="hidden dimension"
						   )
	argparser.add_argument("--layer",
						   type=str,
						   default="rcnn",
						   help="type of recurrent layer"
						   )
	argparser.add_argument("--depth",
						   type=int,
						   default=1,
						   help="number of layers"
						   )
	argparser.add_argument("--pooling",
						   type=int,
						   default=0,
						   help="whether to use mean pooling or the last state"
						   )
	argparser.add_argument("--order",
						   type=int,
						   default=2,
						   help="feature filter width"
						   )
	argparser.add_argument("--use_all",
						   type=int,
						   default=0,
						   help="whether to use the states of all layers"
						   )
	argparser.add_argument("--max_len",
						   type=int,
						   default=-1,
						   help="max number of words in input"
						   )
	argparser.add_argument("--sparsity",
						   type=float,
						   default=0.0003
						   )
	argparser.add_argument("--coherent",
						   type=float,
						   default=2.0
						   )
	argparser.add_argument("--aspect",
						   type=int,
						   default=-1
						   )
	argparser.add_argument("--beta1",
						   type=float,
						   default=0.9
						   )
	argparser.add_argument("--beta2",
						   type=float,
						   default=0.999
						   )
	argparser.add_argument("--decay_lr",
						   type=bool,
						   default=False
						   )

	argparser.add_argument("--reset_params",
						   type=bool,
						   default=False
						   )

	argparser.add_argument("--min_lr",
						   type=float,
						   default=1.0e-6
						   )

	argparser.add_argument("--fix_emb",
						   type=int,
						   default=1
						   )

	argparser.add_argument("--contrastive_z_loss",
						   type=int,
						   default=0,
						   help='If this is on, then instead of regularizing the generator in the normal way, we instead encourage its inverse to be of the opposite class'
						   )
	argparser.add_argument("--prediction_loss_weight",
						   type=float,
						   default=1.0,
						   help='How to weight the prediction loss in the cost function'
						   )
	argparser.add_argument("--inverse_encoder_prediction_loss_weight",
						   type=float,
						   default=0,
						   help='If contrastive z-loss is on, this is how much this term should be weighted'
						   )

	argparser.add_argument("--inverse_generator_prediction_loss_weight",
						   type=float,
						   default=0,
						   help='If contrastive z-loss is on, this is how much this term should be weighted'
						   )

	argparser.add_argument("--z_sparsity_loss_weight",
						   type=float,
						   default=0,
						   help='If contrastive z-loss is on, this is how much the size of the rationale should be penalized'
						   )

	argparser.add_argument("--coherence_loss_weight",
						   type=float,
						   default=0,
						   help='If contrastive z-loss is on, this is how much the size of the rationale should be penalized'
						   )

	argparser.add_argument("--gini_impurity_weight",
						   type=float,
						   default=0,
						   help='Punishes gini impurity in z probabilities. More relevant in soft-attention mode where z = z probs'
						   )

	argparser.add_argument("--retrieval",
						   type=str,
						   default='final_h',
						   help='What retrieval method to use. Currently includes h_final, rationale_centroid, rationale_bigram_centroid and output_weighted_rationale_centroid'
						   )

	argparser.add_argument("--output_layer_bias",
						   type=float,
						   default=None,
						   help='What value to use as an initial value for the bias term of the output layer for the prediction model. This essentially defines what you want the model to do'
								'in the absence of any information (i.e. what to do if the rationale obscures the whole text)'
						   )

	argparser.add_argument("--max_unchanged",
						   type=int,
						   default=10,
						   help='How many consecutive epochs a model can experience no improvement before the training process quits'
						   )

	argparser.add_argument("--num_policy_samples",
						   type=int,
						   default=1,
						   help='How many times we should sample in estimating the rationale policy gradients'
						   )

	argparser.add_argument("--fix_bias_term",
						   type=bool,
						   default=None,
						   help='Whether or not to fix the bias term at some value.'
						   )
	argparser.add_argument("--use_confusion",
						   type=bool,
						   default=True,
						   help='Whether to try to confuse the secondary encoder during training'
						   )
	argparser.add_argument("--use_primary_confusion",
						   type=bool,
						   default=False,
						   help='Whether to try to confuse the primary encoder during training'
						   )
	argparser.add_argument("--bidrectional_generator",
						   type=bool,
						   default=True,
						   help='Whether the generator should be bidirectional or not'
						   )


	argparser.add_argument("--fix_pretraining_bias_term",
						   type=bool,
						   default=None,
						   help='Whether or not to fix the bias term of the pretrained model at some value'
						   )

	argparser.add_argument("--bias_term_fix_value",
						   type=str,
						   default=None,
						   help='What value to fix bias term at. Possible values include False, "mean" for mean y value of training set, or a float'
						   )

	argparser.add_argument("--encoder_pretraining_epochs",
						   type=int,
						   default=0,
						   help='How many epochs to pretrain the encoder for'
						   )

	argparser.add_argument("--tolerance",
						   type=float,
						   default=0.005,
						   help='How much improvement an epoch has to have to be considered an improvement'
						   )

	argparser.add_argument("--dependent",
						   type=bool,
						   default=False,
						   help='Whether to use the version of the algorithm where each zi is dependent or independent of the sampled value of zi-1'
						   )


	argparser.add_argument("--joint_training",
						   type=bool,
						   default=True,
						   help='Whether to jointly or alternatively train the encoder and generator'
						   )

	argparser.add_argument("--encoder_pretrain_epochs",
						   type=int,
						   default=-1,
						   help='How many epochs to pretrain the encoder for, if any'
						   )

	argparser.add_argument("--coherence_method",
						   type=str,
						   default='zdiff',
						   help='How to encourage the raitonale to be more globby. Either zdiff or p0g1'
						   )

	argparser.add_argument("--split_encoder",
						   type=bool,
						   default=True,
						   help='''Whether or not to split the encoder into two models; one which tries to do well on the rationale and one which tries to
								do well on the antirationale'''
						   )

	argparser.add_argument("--hard_attention",
						   type=bool,
						   default=True,
						   help='''Whether the rationale whould be implemented as a hard attention mechanism via stochastic semilinear units and a REINFORCE optimization
						   algorith, or as soft attention. '''
						   )

	argparser.add_argument("--inverse_generator_loss_type",
						   type=str,
						   default='zero',
						   help='''What metric should be used to punish the generator on the encoder's performance on the inverse rationale. 'error', 'zero', 'diff' '''
						   )

	argparser.add_argument("--confusion_method",
						   type=str,
						   default='flip',
						   help='''What additional information should be shown to the secondary encoder to confuse it. Options include: 'shuffle', 'inverse', 'flip' '''
						   )


	argparser.add_argument("--rng_seed",
						   type=int,
						   default=123,
						   help='''Seed to use for secondary encoder mask shuffling'''
						   )
	argparser.add_argument("--original_model",
						   type=bool,
						   default=False,
						   help='''Whether to use the original unmodified model code'''
						   )

	argparser.add_argument("--subsample_training",
						   type=int,
						   default=-1,
						   help='''How to subsample the training set'''
						   )

	argparser.add_argument("--sparsity_method",
						   type=str,
						   default='l1_sum',
						   help='''How to measure sparsity of the rationale mask'''
						   )

	argparser.add_argument("--output_distribution",
						   type=bool,
						   default=False,
						   help='''Whether to output a distribution rather than a value'''
						   )


	argparser.add_argument("--output_distribution_interpretation",
						   type=str,
						   default='regression',
						   help='''How to convert from input target values to a target distribution. In the 'one_hot' case we construct a one-hot vector with a 1.0 value in the appropriate bucket. In the 'class_probability' case we treat the target value as the probability of the positive class'''
						   )
	# args = argparser.parse_args()
	args = set_defaults(argparser)
	return args

def set_defaults(argparser):
	'''
	Create a Namespace object with all default values instead of reading from command line
	:param argparser:
	:return:
	'''
	namespace = argparse.Namespace()

	for action in argparser._actions:
		if action.dest is not argparse.SUPPRESS:
			if not hasattr(namespace, action.dest):
				if action.default is not argparse.SUPPRESS:
					setattr(namespace, action.dest, action.default)

	return namespace



