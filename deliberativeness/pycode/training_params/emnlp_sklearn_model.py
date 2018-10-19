
'''
This is the "sigmoid predictor + feature importance" model from the paper. It doesn't save the model after training because it is so quick.
'''

labeled_sets = [
	{
		'name':'Personal attacks Wiki',
		'prefix':'wiki_attack',
		'train':'../../data/processed/wiki/personal_attacks/wiki_attack_train.csv',
		'dev':'../../data/processed/wiki/personal_attacks/wiki_attack_dev.csv',
		'test':'../../data/processed/wiki/personal_attacks/wiki_attack_test.csv',
		'dev_rationale':'../../data/processed/wiki/personal_attacks/wiki_attack_dev_rationale.csv',
		'test_rationale':'../../data/processed/wiki/personal_attacks/wiki_attack_test_rationale.csv',
	}
]


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
	'inverse_generator_prediction_loss_weight': 0.0,  # recall
	'prediction_loss_weight':1.0,
	'z_sparsity_loss_weight':0.0006,
	'coherence_loss_weight': 2.0,  # recall
	'encoder_pretraining_epochs': 0,
	'fix_bias_term': False,
	'fix_pretraining_bias_term': False,
	'bias_term_fix_value':0.05,
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
	'model_mode':['sklearn'],
	'sklearn_l1_C':[1.0],
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

#Parameter combinations to exclude:
exclude_params = [

]
