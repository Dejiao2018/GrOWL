'''
Code for reproduce results in table 1
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import yaml

from utils_plot import imagesc
import matplotlib
import matplotlib.pyplot as plt
from mnist import train_mnist

NUM_LAYER = 2
log_dir = '../../experiment_results/paper_results/'

def weight_decay():

    with open('./experiment_configs/10_20_config_search.yaml', 'r+') as f_search:
        config = yaml.load(f_search)

        config['use_growl'] = True
        config['growl_params']= [[1e-20, 1e-20], [3e-1, 3e-3]]
        config['use_group_lasso'] = False
        config['use_wd'] = True
        config['log_dir']= log_dir
        lr = 1e-3
        config['learning_rate'] = lr
        # for baseline
        config['num_epochs'] = 300
        config['retraining'] = True

        wd = [0.01] * 5
        for i, wd_i in enumerate(wd):
            config['wd'] = wd[i]
            config['res_dir'] = config['log_dir'] + '/weight_decay_{}_p_{}_idx_{}'.format(wd_i, config['preference'], i)
            config['plot_dir'] = config['res_dir'] + '/plot_res'
            config['summary_dir'] = config['res_dir'] + '/summary_res'
            train_mnist(config)

def baseline():

    with open('./experiment_configs/10_20_config_search.yaml', 'r+') as f_search:
        config = yaml.load(f_search)

        config['use_growl'] = False
        config['growl_params']= [[1e-20, 1e-20], [3e-1, 3e-3]]
        config['use_group_lasso'] = False
        config['use_wd'] = False
        config['log_dir']= log_dir
        lr = 1e-3
        config['learning_rate'] = lr
        # for baseline
        config['num_epochs'] = 400
        config['retraining'] = False

        for i in range(5):
            config['res_dir'] = config['log_dir'] + '/baseline_idx_{}'.format(i)
            config['plot_dir'] = config['res_dir'] + '/plot_res'
            config['summary_dir'] = config['res_dir'] + '/summary_res'
            train_mnist(config)

def group_lasso():

    with open('./experiment_configs/10_20_config_search.yaml', 'r+') as f_search:
        config = yaml.load(f_search)

        config['use_growl'] = False
        config['use_group_lasso'] = True
        config['use_wd'] = False
        lr = 1e-3
        config['learning_rate'] = lr
        config['log_dir']= log_dir

        lambda1 = [60] * 5

        for idx1, lambda1_i in enumerate(lambda1):
            config['growl_params'] = [[lambda1_i, 0]] * NUM_LAYER
            config['res_dir'] = config['log_dir'] + '/group_lasso_lr_{}_l1_p_{}_{}_idx1_{}'.format(lr, lambda1_i, config['preference'], idx1)
            config['plot_dir'] = config['res_dir'] + '/plot_res'
            config['summary_dir'] = config['res_dir'] + '/summary_res'
            train_mnist(config)

def PLD():

    with open('./experiment_configs/10_20_config_search.yaml', 'r+') as f_search:
        config = yaml.load(f_search)

        config['use_growl'] = True
        config['use_group_lasso'] = False
        config['use_wd'] = False
        config['reg_params_type'] = 'PLD'

        PLD_transition = 0.5
        preference = 0.6
        lr = 1e-3
        config['learning_rate'] = lr
        config['log_dir']= log_dir

        lambda1 = 87

        lambda1 = [12] * 5
        lambda2 = [0.215]
        lambda_wd = [0]

        for idx1, lambda1_i in enumerate(lambda1):
            for lambda2_i in lambda2:
                for wd in lambda_wd:
                    config['preference'] = preference
                    config['wd'] = wd
                    config['growl_params'] = [[lambda1_i, lambda2_i]] * NUM_LAYER
                    config['PLD_transition'] = PLD_transition
                    config['res_dir'] = config['log_dir'] + '/PLDwd_lr_{}_l1_{}_l2_{}_PLDt_{}_p_{}_wd_{}_idx_{}'.format(lr, lambda1_i, lambda2_i, PLD_transition, config['preference'], wd, idx1)
                    config['plot_dir'] = config['res_dir'] + '/plot_res'
                    config['summary_dir'] = config['res_dir'] + '/summary_res'
                    train_mnist(config)

def PLD_l2():

    with open('./experiment_configs/10_20_config_search.yaml', 'r+') as f_search:
        config = yaml.load(f_search)

        config['use_growl'] = True
        config['use_group_lasso'] = False
        config['use_wd'] = True
        config['reg_params_type'] = 'PLD'

        PLD_transition = 0.5
        preference = 0.6
        lr = 1e-3
        config['learning_rate'] = lr
        config['log_dir']= log_dir

        lambda1 = [12] * 5 
        lambda2 = [0.235]
        lambda_wd = [0.001]

        for idx1, lambda1_i in enumerate(lambda1):
            for lambda2_i in lambda2:
                for wd in lambda_wd:
                    config['preference'] = preference
                    config['wd'] = wd
                    config['growl_params'] = [[lambda1_i, lambda2_i]] * NUM_LAYER
                    config['PLD_transition'] = PLD_transition
                    config['res_dir'] = config['log_dir'] + '/PLDwd_lr_{}_l1_{}_l2_{}_PLDt_{}_p_{}_wd_{}_idx_{}'.format(lr, lambda1_i, lambda2_i, PLD_transition, config['preference'], wd, idx1)
                    config['plot_dir'] = config['res_dir'] + '/plot_res'
                    config['summary_dir'] = config['res_dir'] + '/summary_res'
                    train_mnist(config)

def grlasso_l2():

    with open('./experiment_configs/10_20_config_search.yaml', 'r+') as f_search:
        config = yaml.load(f_search)

        config['use_growl'] = False
        config['use_group_lasso'] = True
        config['use_wd'] = True

        lr = 1e-3
        config['learning_rate'] = lr

        config['log_dir']= log_dir

        lambda1 = [80] * 5
        lambda_wd = [0.001]
        preference = [0.6]
        
        for p in preference:
            config['preference'] = p
            for wd in lambda_wd:
                for idx1, lambda1_i in enumerate(lambda1):
                    config['growl_params'] = [[lambda1_i, 0]] * NUM_LAYER
                    config['wd'] = wd
                    config['res_dir'] = config['log_dir'] + '/grEN_l1_{}_l_wd_{}_p_{}_idx_{}'.format(lambda1_i, wd, config['preference'], idx1)
                    config['plot_dir'] = config['res_dir'] + '/plot_res'
                    config['summary_dir'] = config['res_dir'] + '/summary_res'
                    train_mnist(config)


if __name__ == "__main__":

    PLD()
    PLD_l2()
    
    group_lasso()
    grlasso_l2()
    
    weight_decay()
    baseline()