from modeling.train_and_evaluate_models import synthetic_with_neutral_rationale_function, synthetic_rationale_function

'''
A couple of synthetic datasets where the texts consist just of the words "good" and "bad" (and "neutral" in the second set).

'''


labeled_sets = [

{
		'name': 'Synthetic with OR target',
		'prefix': 'synthetic_or_target',
		'train': '../../data/processed/synthetic/or_target/or_target_train.csv',
		'dev': '../../data/processed/synthetic/or_target/or_target_dev.csv',
		'test': '../../data/processed/synthetic/or_target/or_target_test.csv',
		'dev_rationale':synthetic_rationale_function,
		'test_rationale':synthetic_rationale_function,
	'embeddings':'../../data/processed/synthetic/embeddings.txt.gz'

	},

{
		'name': 'Synthetic with neutral words',
		'prefix': 'synthetic_neutral_target',
		'train': '../../data/processed/synthetic/neutral_target/neutral_target_train.csv',
		'dev': '../../data/processed/synthetic/neutral_target/neutral_target_dev.csv',
		'test': '../../data/processed/synthetic/neutral_target/neutral_target_test.csv',
		'dev_rationale':synthetic_with_neutral_rationale_function,
		'test_rationale':synthetic_with_neutral_rationale_function,
		'embeddings':'../../data/processed/synthetic/embeddings.txt.gz',
	}]



#These are unlabeled datasets
unlabeled_sets = [
]




all_params = {
	'max_unchanged':10,
	'activation':'tanh',
	'batch': 250,
	'embedding_size':200,
	'use_confusion':True,
	'use_primary_confusion':False,
	'inverse_generator_prediction_loss_weight': [0.0, 1.0],  # recall
	'prediction_loss_weight':1.0,
	'z_sparsity_loss_weight':[0.0015],
	'coherence_loss_weight': [0.0],  # recall
	'encoder_pretraining_epochs': 0,
	'fix_bias_term': False,
	'fix_pretraining_bias_term': False,
	'bias_term_fix_value':[0.05],
	'generator_bias_term_fix_value':False,
	'sparsity_method': 'l1_sum',
	'coherence_method': 'zdiff_sum',
	'split_encoder': True,
	'inverse_generator_loss_type': 'zero',
	'confusion_method':'flip',
	'learning_rate': 0.0005,
	'max_epochs':1,
	'hidden_dimension': 200,
	'layer': 'rcnn',
	'l2_reg': 1e-6,
	'depth':2,
	'max_len':256,
	'use_all':True,
	'return_c':False,
	'bidirectional_generator': True,
	'dependent': True,
	'hard_attention': True,
	'fix_rationale_value': None,
	'decay_lr':True,
	'reset_params': True,
	'num_policy_samples':1,
	'eval_period':3, #save model every 3 useful epochs,
	'joint_training':True,
	'tolerance':0.1,
	'dropout':0.1,
	'pooling':False,
	'rng_seed':9912,

}

exclude_params = [
]