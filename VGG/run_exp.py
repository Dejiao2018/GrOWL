from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import sys
import six
from shutil import copyfile

import cifar_input
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from flags import FLAGS
from utils_GrOWL import reg_params_init, apply_reg_prox, update_mask, measure_compression, preprocess_hparam, adjust_learning_rate, set_param_share
from utils_retrain import display_similarity, apply_param_share, group_averaging
from vgg_model import vgg16
from vgg_main import train, retrain

BATCH_PER_EPOCH = int(FLAGS.num_training_examples_per_epoch / FLAGS.batch_size)
TRAIN_STEPS = BATCH_PER_EPOCH * FLAGS.num_training_epochs
RETRAIN_STEPS = BATCH_PER_EPOCH * FLAGS.num_retraining_epochs
EVAL_BATCHES = int(10000/100)
REG_APPLY_FREQ = FLAGS.reg_apply_epoch_freq * BATCH_PER_EPOCH

def main(argv=None):

    batch_size = 128
    num_classes = 10

    # lambda1_scale = [0.6, 0.8, 1.0, 1.2]
    lambda1_scale = [1.0]
    lambda2_scale = [1]
    preference_list = [0.8]
    for prefer_val in preference_list:

        if prefer_val == 0.8:
            idx=1
        else:
            idx=0

        for lambda1_scale_i in lambda1_scale:

            for lambda2_scale_i in lambda2_scale:
                idx += 1
                FLAGS.load_root = '../VGGres_revise/12_20_GrOWL_EN/search/res_08_{}/'.format(idx)
                FLAGS.log_root = '../VGGres_revise/12_20_GrOWL_EN/search/res_0{}_{}/'.format(int(prefer_val*10),idx)
                # FLAGS.train_dir_load = FLAGS.load_root + 'train_dir/'
                FLAGS.train_dir_load = FLAGS.load_root + 'train_dir/'

                FLAGS.eval_dir = FLAGS.log_root + 'eval_dir/'
                FLAGS.train_dir = FLAGS.log_root + 'train_dir/'
                FLAGS.retrain_dir = FLAGS.log_root + 'retrain_dir/'
                FLAGS.plot_dir = FLAGS.log_root + 'plot_dir/'
                FLAGS.res_dir = FLAGS.log_root + 'res_dir/'
                FLAGS.preference = prefer_val
                
                
                reg_params=np.array([[100, 100],[4.5, 1e-1],[11, 1e-1],[11, 1e-1],[11, 1e-1],
                                [10, 1e-1],[9, 1e-1],[7.5, 1e-1],[7.0, 1e-1],[6.5, 7e-2],
                                [5.5, 5e-2],[5.5, 1e-1],[3.5, 1e-1],[3.5, 1e-1],[100,100]])

                reg_params[:,0] = reg_params[:,0] * lambda1_scale_i
                reg_params[:,1] = reg_params[:,1] * lambda2_scale_i

                # 1-3, and 15 no reg, 4-14 reg
                reg_applied_layers = [False, True, True, True, True,
                                      True, True, True, True, True,
                                      True, True, True, True, False]

                hps, res_dict=preprocess_hparam(batch_size=batch_size,
                                         num_classes=num_classes,
                                         reg_params=reg_params,
                                         reg_applied_layers=reg_applied_layers,
                                         param_shared_layers=reg_applied_layers)
                #Save configuration info
                copyfile('flags.py', FLAGS.log_root + 'flags.py')
                try:
                    zero_layers, step = train(hps, res_dict)
                    if zero_layers and step <= 390*60:
                        continue
                    else:
                        #only enforce parameter sharing to those layers which form the pattern of clusters
                        param_shared_layers = set_param_share(data_dir=FLAGS.res_dir, fileName='cluster.txt', 
                                                      param_shared_layers=reg_applied_layers)

                        hps, res_dict=preprocess_hparam(batch_size=batch_size,
                                                 num_classes=num_classes,
                                                 reg_params=reg_params,
                                                 reg_applied_layers=reg_applied_layers,
                                                 param_shared_layers=param_shared_layers)
                        retrain(hps, res_dict)

                except KeyboardInterrupt:
                    break

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
