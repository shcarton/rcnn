Each python file in here is a parameter file for the cotrain_models script. They are python files instead of json for increased flexibility (e.g. specifying functions instead of files for certain kinds of inputs)

The way these files are structured is basically a janky, home-brewed machine learning experiment specification scheme, where a single file can represent a series of labeled sets, unlabeled sets and parameter combinations to test out and compare to one another.

Each file consists of 4 variables: labeled_sets, unlabeled_sets, all_params, and exclude_params

labeled_sets is a list of dictionaries where each dictionary decribes a labeled dataset to train and evaluate one or more models on. You specify the paths of the train/dev/test sets, the (separate) rationale files if they exist, and you can also specify the location of an embedding file to use. If such an embedding file does not exist, it will be created automatically by running word2vec on the training set and will be saved to the output directory for that labeled set.

There are certain columns that a labeled set should have. Check out data/processed/wiki/personal_attacks/wiki_attack_train.csv to see what these are.


unlabeled_sets is a list of dictionaries representing unlabeled datasets to run the model over after it has been trained. Instead of train/dev/test, the dictionary for this should just have a "file" key-value pair pointing to a csv. Otherwise it shouldbe similar to labeled_sets.



all_params is a dictionary of parameter values whose meanings and default values are defined in s_options.py. There are a zillion of these, as I've experimented with different model structures, attention mechanisms, sparsity methods, etc. When you want to try multiple combinations of parameters, you can specify a list of values instead of a single value, and train_and_evaluate_models.py will automatically try all unique combinations of list-valued parameters. You can exclude any parameter combination you don't want to try using the exclude_params data structure below.

A model trained on a set of parameters is given a name depending on which parameters are list-valued. For example, the model produced by the param set in best_emnlp_model will be named "btfv=0.05_bg=True_clw=2.0_cm=zdiff_sum_cm=flip_d=True_fbt=False_fpbt=False_ha=True_iglt=zero_igplw=1.0_jt=True_l=rcnn_sm=l1_sum_se=True_uc=True_upc=False_zslw=0.0015", because those (abbreviated) parameters are the ones that are given list values in the all_params for that config. Whenever you try to train a model with the same named-parameter profile, the training script will automatically try to load any existing model with the same profile, unless you turn off this behavior by setting "load_existing_model" to false in that script.


exclude_params is a list of dictionaries indicating parameter combinations to exclude. It will exclude any parameter combination that matches a specified dictionary. For example, the best_emnlp_lei_variants.py all_params dictionary calls for 2x2x2 = 8 different parameter combinations, but it excludes 5 of those possible combinations by specifying three prohibited combinations of parameter values in the exclude_params list.