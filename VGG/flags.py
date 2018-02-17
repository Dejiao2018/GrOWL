from __future__ import division
import os
import sys
import tensorflow as tf
import numpy as np
from collections import namedtuple

flags= tf.app.flags
FLAGS = flags.FLAGS

HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'use_weight_decay, weight_decay_rate, '
                     'relu_leakiness, optimizer, reg_applied_layers,'
                     'reg_params, param_shared_layers')


log_root = '../../VGG_result/'
eval_dir = log_root + 'eval_dir/'
train_dir = log_root + 'train_dir/'
retrain_dir = log_root + 'retrain_dir/'
plot_dir = log_root + 'plot_dir/'
res_dir = log_root + 'res_dir/'


flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
flags.DEFINE_integer('image_size', 32, 'Image side length.')

flags.DEFINE_bool('resume', False,
                    'Whether resume an existed model or not.')
flags.DEFINE_float('resume_lr_init', 0.001,
                    'resume learning rate')

flags.DEFINE_string('mode', 'train', 'train or eval.')

flags.DEFINE_string('train_data_path', '../datasets/cifar_10/data_batch_*.bin',
                           'Filepattern for training data.')
flags.DEFINE_integer('num_training_examples_per_epoch', 50000,
                           'Number of training images')

flags.DEFINE_string('eval_data_path', '../datasets/cifar_10/test_batch.bin',
                           'Filepattern for eval data')
flags.DEFINE_integer('num_testing_examples_per_epoch', 10000,
                           'Number of testing images')

#Train_dir, Eval_dir and Summary_dir
flags.DEFINE_string('train_dir', train_dir,
                           'Directory to keep training outputs.')
flags.DEFINE_string('retrain_dir', retrain_dir,
                           'Directory to store retraining models.')
flags.DEFINE_string('eval_dir', eval_dir,
                           'Directory to keep eval outputs.')
flags.DEFINE_string('plot_dir', plot_dir,
                           'Directory to keep plots.')
flags.DEFINE_string('res_dir', res_dir,
                           'Directory to keep results.')
flags.DEFINE_string('log_root', log_root,
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')

#Training-retraining
flags.DEFINE_boolean('retrain_on', True,
    'turn on or turn off the retraining process')
flags.DEFINE_boolean('param_share', True,
    'enforce parameter sharing or not')
flags.DEFINE_float('preference', 0.8,
                   'preference value for affinity propagation')


flags.DEFINE_integer('batch_size', 128,
                            'Batch size.')
flags.DEFINE_integer('num_training_epochs', 15,
                            'number of training epochs.')
flags.DEFINE_integer('num_retraining_epochs',5,
                            'number of retraining epochs.')
flags.DEFINE_float('training_lr_init', 1e-2,
                         'initial learning rate for the training process')
flags.DEFINE_float('retraining_lr_init', 1e-3,
                        'initial learning rate for the retraining process')
flags.DEFINE_float('lr_decay_rate', 0.1,
                        'learning rate decay factor')
flags.DEFINE_float('lr_decay_step', 6*390, "decay the learning rate every several steps")
flags.DEFINE_float('retrain_lr_decay_step', 2*390, "decay the learning rate every several steps")


flags.DEFINE_integer('eval_batch_count', 100,
                            'Number of batches to eval.')
flags.DEFINE_bool('eval_once', True,
                         'Whether evaluate the model only once.')

#L2 regularization
flags.DEFINE_boolean('use_weight_decay', True,
                  'use l2 weight decay or not')
flags.DEFINE_float('weight_decay_rate', 2e-3,
                        'weight decay coefficients')

#Sparsity inducing regularization
flags.DEFINE_boolean('use_sparse_reg', True,
                  'use sparsity inducing regularizers (owl, growl, group Lasso, l1) or not')
flags.DEFINE_integer('reg_apply_epoch_freq', 1,
                      'regularization applied frequency, defined in terms of epochs')
flags.DEFINE_boolean('use_mask', True,
                  'use masks to prevent drifting from zeros')
flags.DEFINE_float('mask_threshold', 1e-7,
                    'threshold value, for pruning extremely small rows')


flags.DEFINE_boolean('use_growl', True,
                     'use group OWL or not')
flags.DEFINE_boolean('use_group_lasso', False,
                     'use group Lasso or not')

flags.DEFINE_string('PLD_transition', 0.1, '[0,1]')



#Display, Save, Summary frequencies
flags.DEFINE_integer('display_similarity_freq', 390*2,
                     'frequency of displaying the similarity plots of each layer')
flags.DEFINE_integer('tensorboard_freq', 390,
                     'frequency of writing summaries to tensorboard')
flags.DEFINE_integer('checkpoint_freq', 390*4,
                     'frequency of saving model')
flags.DEFINE_integer('checkpoint_freq_retrain', 390*2,
                     'frequency of saving model')
flags.DEFINE_integer('row_norm_freq', 390*2, 'frequency of writing the histogram of the row norms')
flags.DEFINE_integer('eval_freq', 390, 'frequency of evaluation')
