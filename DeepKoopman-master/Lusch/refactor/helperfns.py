import datetime
import pickle
import time

import numpy as np
# import tensorflow as tf


def set_defaults(params):
    """Set defaults and make some checks in parameters dictionary.

    Arguments:
        params -- dictionary of parameters for experiment

    Returns:
        None (but side effect of updating params dict)

    Side effects:
        May update params dict

    Raises KeyError if params is missing data_name, len_time, data_train_len, delta_t, widths, hidden_widths_omega,
        num_evals, num_real, or num_complex_pairs
    Raises ValueError if num_evals != 2 * num_complex_pairs + num_real
    """
    # defaults related to dataset
    if 'data_name' not in params:
        raise KeyError("Error: must give data_name as input to main")
    if 'len_time' not in params:
        raise KeyError("Error, must give len_time as input to main")
    if 'data_train_len' not in params:
        raise KeyError("Error, must give data_train_len as input to main")
    if 'delta_t' not in params:
        raise KeyError("Error, must give delta_t as input to main")

    # defaults related to saving results
    if 'folder_name' not in params:
        # print("setting default: using folder named 'results'")
        params['folder_name'] = 'results'
    if 'exp_suffix' not in params:
        # print("setting default name of experiment")
        params['exp_suffix'] = '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if 'model_path' not in params:
        # print("setting default path for model")
        exp_name = params['data_name'] + params['exp_suffix']
        params['model_path'] = "./%s/%s_model.ckpt" % (params['folder_name'], exp_name)

    # defaults related to network architecture
    if 'widths' not in params:
        raise KeyError("Error, must give widths as input to main")
    # print(params['widths'])
    if 'hidden_widths_omega' not in params:
        raise KeyError("Error, must give hidden_widths for omega net")
    params['widths_omega_complex'] = [1, ] + params['hidden_widths_omega'] + [2, ]
    params['widths_omega_real'] = [1, ] + params['hidden_widths_omega'] + [1, ]
    # print(params['widths_omega_complex'])  
    # print(params['widths_omega_real'])

    if 'act_type' not in params:
        # print("setting default: activation function is ReLU")
        params['act_type'] = 'relu'

    if 'num_evals' not in params:
        raise KeyError("Error, must give number of evals: num_evals")
    if 'num_real' not in params:
        raise KeyError("Error, must give number of real eigenvalues: num_real")
    if 'num_complex_pairs' not in params:
        raise KeyError("Error, must give number of pairs of complex eigenvalues: num_complex_pairs")
    # if params['num_evals'] != (2 * params['num_complex_pairs'] + params['num_real']):
    #     raise ValueError("Error, num_evals must equal 2*num_compex_pairs + num_real")

    params['d'] = len(params['widths'])  # d must be calculated like this

    # defaults related to initialization of parameters
    if 'seed' not in params:
        random_seed = np.random.randint(2 ** 30)
        # print("setting default: choosing random seed of %d and saving to params" % random_seed)
        params['seed'] = random_seed
    if 'dist_weights' not in params:
        # print("setting default: distribution for weights on main net is tn (truncated normal)")
        params['dist_weights'] = 'tn'
    if 'dist_weights_omega' not in params:
        # print("setting default: distribution for weights on auxiliary net is tn (truncated normal)")
        params['dist_weights_omega'] = 'tn'
    if 'dist_biases' not in params:
        # print("setting default: biases in main net will be init. to default number")
        params['dist_biases'] = 0
    if 'dist_biases_omega' not in params:
        # print("setting default: biases in auxiliary net will be init. to default number")
        params['dist_biases_omega'] = 0

    if 'scale' not in params:
        # print("setting default: scale for weights in main net is 0.1 (applies to tn distribution)")
        params['scale'] = 0.1
    if 'scale_omega' not in params:
        # print("setting default: scale for weights in omega net is 0.1 (applies to tn distribution)")
        params['scale_omega'] = 0.1

    if isinstance(params['dist_weights'], str):
        params['dist_weights'] = [params['dist_weights']] * (len(params['widths']) - 1)
    if isinstance(params['dist_biases'], int):
        params['dist_biases'] = [params['dist_biases']] * (len(params['widths']) - 1)
    if isinstance(params['dist_weights_omega'], str):
        params['dist_weights_omega'] = [params['dist_weights_omega']] * (len(params['widths_omega_real']) - 1)
    if isinstance(params['dist_biases_omega'], int):
        params['dist_biases_omega'] = [params['dist_biases_omega']] * (len(params['widths_omega_real']) - 1)

    # defaults related to loss function
    if 'auto_first' not in params:
        params['auto_first'] = 0
    if 'relative_loss' not in params:
        # print("setting default: loss is not relative")
        params['relative_loss'] = 0

    if 'shifts' not in params:
        # print("setting default: penalty on all shifts from 1 to num_shifts")
        params['shifts'] = np.arange(params['num_shifts']) + 1
    if 'shifts_middle' not in params:
        # print("setting default: penalty on all middle shifts from 1 to num_shifts_middle")
        params['shifts_middle'] = np.arange(params['num_shifts_middle']) + 1
    params['num_shifts'] = len(params['shifts'])  # must be calculated like this
    params['num_shifts_middle'] = len(params['shifts_middle'])  # must be calculated like this

    if 'recon_lam' not in params:
        # print("setting default: weight on reconstruction is 1.0")
        params['recon_lam'] = 1.0
    if 'mid_shift_lam' not in params:
        # print("setting default: weight on loss3 is 1.0")
        params['mid_shift_lam'] = 1.0
    if 'L1_lam' not in params:
        # print("setting default: L1_lam is .00001")
        params['L1_lam'] = .00001
    if 'L2_lam' not in params:
        # print("setting default: no L2 regularization")
        params['L2_lam'] = 0.0
    if 'Linf_lam' not in params:
        # print("setting default: no L_inf penalty")
        params['Linf_lam'] = 0.0

    # defaults related to training
    if 'num_passes_per_file' not in params:
        # print("setting default: 1000 passes per training file")
        params['num_passes_per_file'] = 1000
    if 'num_steps_per_batch' not in params:
        # print("setting default: 1 step per batch before moving to next training file")
        params['num_steps_per_batch'] = 1
    if 'num_steps_per_file_pass' not in params:
        # print("setting default: up to 1000000 steps per training file before moving to next one")
        params['num_steps_per_file_pass'] = 1000000
    if 'learning_rate' not in params:
        # print("setting default learning rate")
        params['learning_rate'] = .003
    if 'opt_alg' not in params:
        # print("setting default: use Adam optimizer")
        params['opt_alg'] = 'adam'
    if 'decay_rate' not in params:
        # print("setting default: decay_rate is 0 (applies to some optimizer algorithms)")
        params['decay_rate'] = 0
    if 'batch_size' not in params:
        # print("setting default: no batches (use whole training file at once)")
        params['batch_size'] = 0

    # setting defaults related to keeping track of training time and progress
    if 'max_time' not in params:
        # print("setting default: run up to 6 hours")
        params['max_time'] = 6 * 60 * 60  # 6 hours
    if 'min_5min' not in params:
        params['min_5min'] = 10 ** (-2)
        # print("setting default: must reach %f in 5 minutes" % params['min_5min'])
    if 'min_20min' not in params:
        params['min_20min'] = 10 ** (-3)
        # print("setting default: must reach %f in 20 minutes" % params['min_20min'])
    if 'min_40min' not in params:
        params['min_40min'] = 10 ** (-4)
        # print("setting default: must reach %f in 40 minutes" % params['min_40min'])
    if 'min_1hr' not in params:
        params['min_1hr'] = 10 ** (-5)
        # print("setting default: must reach %f in 1 hour" % params['min_1hr'])
    if 'min_2hr' not in params:
        params['min_2hr'] = 10 ** (-5.25)
        # print("setting default: must reach %f in 2 hours" % params['min_2hr'])
    if 'min_3hr' not in params:
        params['min_3hr'] = 10 ** (-5.5)
        # print("setting default: must reach %f in 3 hours" % params['min_3hr'])
    if 'min_4hr' not in params:
        params['min_4hr'] = 10 ** (-5.75)
        # print("setting default: must reach %f in 4 hours" % params['min_4hr'])
    if 'min_halfway' not in params:
        params['min_halfway'] = 10 ** (-4)
        # print("setting default: must reach %f in first half of time allotted" % params['min_halfway'])

    # initializing trackers for how long the training has run
    params['been5min'] = 0
    params['been20min'] = 0
    params['been40min'] = 0
    params['been1hr'] = 0
    params['been2hr'] = 0
    params['been3hr'] = 0
    params['been4hr'] = 0
    params['beenHalf'] = 0

