# -*- coding: utf-8 -*-
'''
This modules contains functions necessary for applying OWL or group OWL
to the parameters
  1. reg_params_init
  2. apply_growl
  3. apply_owl_prox
  4. update_mask
  5. measure_compression
  6. adjust_learning_rate
  7. preprocess_hparams
  8. set_param_share
'''

from __future__ import division, print_function, absolute_import
import sys
sys.path.append('./owl_projection')

import tensorflow as tf
import numpy as np
from projectedOWL import proxOWL
from numpy.linalg import norm
from math import sqrt
from utils_nn import get_weight_placeholders, get_mask_placeholders
from flags import FLAGS, HParams
import re
import os

def reg_params_init(sess, hps):
  '''
  This function initializes the regularization paramters.

  Args:
    sess: the predefined computation graph.
    hps: hyperparameters collection

  Returns:
    layer_owl_params: a list, each element is an array containing the weights
              of the corresponding layer.
  '''
  weight_placeholder = get_weight_placeholders()
  reg_applied_layers = hps.reg_applied_layers

  layer_owl_params = []
  for idx, triple in enumerate(weight_placeholder):

    print('layer {}'.format(idx))
    # if the layer is not regularized, then append []
    if not reg_applied_layers[idx]:
        layer_owl_params.append([])
        continue
    
    #Regularization parameters
    reg_params = hps.reg_params
    lambda_1 = np.float32(reg_params[idx][0])
    lambda_2 = np.float32(reg_params[idx][1])
    if (lambda_1 < 0) | (lambda_2 < 0):
        raise Exception('regularization parameters must be non-negative')

    #GrOWL weights should be applied to the rows of the (reshaped) weight matrix
    param_i, placeholder_i, assign_op_i = triple
    param_shape = sess.run(tf.shape(param_i))

    if np.size(param_i.get_shape().as_list()) == 2:
      row_num = param_shape[0]

    elif np.size(param_i.get_shape().as_list()) == 4:
      row_num = param_shape[2]

    transition_ind = np.floor(row_num*FLAGS.PLD_transition)
    param_index = np.linspace(start=transition_ind-1, stop=0, num=transition_ind)
    print('  row num: {}, transition_ind: {}, largest reg: {}'.format(row_num, transition_ind, lambda_1 + lambda_2 * transition_ind))
    if row_num > transition_ind:
      param_index = np.append(param_index, np.zeros([1, int(row_num-transition_ind)]))

    layer_owl_params.append(lambda_1 + lambda_2 * param_index)

  print("length of weight_placeholder:{0}".format(len(weight_placeholder)))

  assert len(layer_owl_params) == len(weight_placeholder)
  assert len(layer_owl_params) == len(hps.reg_applied_layers)

  return layer_owl_params, hps


def apply_group_lasso(W, weights):

  #Prox op
  W_norm = norm(W, axis=1)
  new_W_norm = np.maximum(W_norm - weights[0], 0)

  new_W = np.zeros_like(W)
  for i in range(W.shape[0]):

    if W_norm[i] < np.finfo(np.float32).eps:
      new_W[i,:] = 0 * W[i,:]
    else:
      new_W[i,:] = new_W_norm[i] * W[i,:] / W_norm[i]

  return new_W


def apply_growl(W, weights):

  # Prox op
  W_norm = norm(W, axis=1)
  new_W_norm=proxOWL(W_norm, weights)
  
  new_W = np.zeros_like(W)
  for i in range(W.shape[0]):
    if W_norm[i] < np.finfo(np.float32).eps:
      new_W[i,:] = 0 * W[i,:]
    else:
      new_W[i,:] = new_W_norm[i] * W[i,:] / W_norm[i]

  return new_W


def apply_reg_prox(sess, learning_rate_val, layer_reg_params, hps):
  '''
  Updates the weights parameter of each layer

  Args:
    sess: the comptutaion graph
    learning_rate: the predefined learning rate
    layer_reg_params: owl parameters, initially created by reg_params_init
    hps:

  Returns:
    None
  '''
  # get weights of the network
  weight_placeholders = get_weight_placeholders()

  # prox_lr_val = min(learning_rate_val, 0.001)
  prox_lr_val = learning_rate_val
  for idx, triple in enumerate(weight_placeholders):

    #Don't apply owl/growl if told not to
    if not hps.reg_applied_layers[idx]:
      continue

    param_i, placeholder_i, assign_op_i = triple
    param_val = sess.run(param_i)
    dim_i = np.size(param_val.shape)

    if dim_i == 2:
      if FLAGS.use_growl:
        prox_param_val = apply_growl(param_val, prox_lr_val * layer_reg_params[idx])
      else:
        prox_param_val = apply_group_lasso(param_val, prox_lr_val * layer_reg_params[idx])

    elif dim_i == 4:
      # For convolutional layer, we need to first reshape 4D tensor to 2D matrix
      reduced_param_val = reshape_2D_4D(param_val, target_shape=None,
                                        reshape_type=2, reshape_order='F')
      if FLAGS.use_growl:
        reduced_prox_param_val = apply_growl(reduced_param_val, prox_lr_val * layer_reg_params[idx])
      else:
        reduced_prox_param_val = apply_group_lasso(reduced_param_val, prox_lr_val * layer_reg_params[idx])

      #Now reshape the 2D matrix back to 4D tensor
      prox_param_val = reshape_2D_4D(reduced_prox_param_val, target_shape=param_val.shape,
                                      reshape_type=1, reshape_order='F')

    # assign the new weights to param_i using the assign_op_i
    sess.run(assign_op_i, feed_dict={placeholder_i:prox_param_val})


def update_mask(sess, threshold, hps, res_dict, step):
  '''
  update the mask during the training process to prevent drifting from zero

  Args:
    sess: the computation graph
    learning_rate: the predefined learning rate
    threshold: the pruning threshold, this may help avoid the floating number error
           occured during the masking process
    model: the resnet class
    hps: hyperparameters
    res_dict: results dictionary
    step: current step

  Returns:
    num_zero_layers: number of zero valued layers
  '''

  mask_palceholders = get_mask_placeholders()
  weight_placeholders = get_weight_placeholders()

  #count the zero valued layers in order to avoiding the nonsense results
  num_zero_layers = 0
  layer_ID = []

  assert len(mask_palceholders) == len(weight_placeholders)


  for idx, mask_triple in enumerate(mask_palceholders):

    #Don't apply owl/growl if told not to
    if not hps.reg_applied_layers[idx]:
      continue

    mask_i, mask_palceholders_i, mask_assign_op_i = mask_triple
    param_i, param_placeholder_i, param_assign_op_i = weight_placeholders[idx]
    dim_i = param_i.get_shape().as_list()

    #Recover the masked weights to zeros if they drifted
    param_val = sess.run(param_i)
    mask = sess.run(mask_i)
    param_val_masked = param_val * mask

    #If apply to convolutional layer, compute the reshaped matrix
    if np.size(dim_i) == 4:
      param_val_masked_reshaped = reshape_2D_4D(param_val_masked, target_shape=None,
                                        reshape_type=2, reshape_order='F')
      mask_reshaped = reshape_2D_4D(mask, target_shape=None,
                                  reshape_type=2, reshape_order='F')

      #prune params and update the mask
      row_norm = norm(param_val_masked_reshaped, axis=1)
      row_size = param_val_masked_reshaped.shape[1]
      print('layer:{}, largest row norm: {:6f}, median row norm: {:.6f}, min row norm: {:.6f}'.format(idx, np.max(row_norm), np.median(row_norm), np.min(row_norm)))

      zero_row_idx = np.where(row_norm <=threshold)
      print('    masked neurons: {}; total neurons: {}'.format(np.size(zero_row_idx), np.size(row_norm)))
      param_val_masked_reshaped[zero_row_idx[0], :] = 0
      mask_reshaped[zero_row_idx[0], :] = 0

      #back to 4D
      param_val_masked = reshape_2D_4D(param_val_masked_reshaped, target_shape=tuple(dim_i),
                                   reshape_type=1, reshape_order='F')
      mask = reshape_2D_4D(mask_reshaped, target_shape=tuple(dim_i),
                              reshape_type=1, reshape_order='F')
    
    elif np.size(dim_i) == 2:
      row_norm = norm(param_val_masked, axis=1)
      row_size = param_val_masked.shape[1]
      print('layer:{}, largest row norm: {:6f}, median row norm: {:.6f}, min row norm: {:.6f}'.format(idx, np.max(row_norm), np.median(row_norm), np.min(row_norm)))

      zero_row_idx = np.where(row_norm <=threshold)
      print('    masked rows: {}; total rows: {}'.format(np.size(zero_row_idx), np.size(row_norm)))
      param_val_masked[zero_row_idx[0], :] = 0
      mask[zero_row_idx[0], :] = 0

    #Update the mask and weight matrix
    sess.run(mask_assign_op_i, feed_dict={mask_palceholders_i:mask})
    sess.run(param_assign_op_i, feed_dict={param_placeholder_i:param_val_masked})

   
    nonzero_rows = np.size(row_norm) - np.size(zero_row_idx[0])
    layer_nonzero_params = nonzero_rows * row_size
    print("    total:{0}, nonzeros:{1}".format(np.size(param_val_masked),
                                               layer_nonzero_params))
    
    ################################
    #Record the zero valued layers
    if np.size(row_norm) - np.size(zero_row_idx[0]) <= 3:
      num_zero_layers += 1
      layer_ID += [idx]

  return num_zero_layers, layer_ID


def measure_compression(sess, res_dict, step, training, hps, num_cluster_arr=[]):
  '''
  Monitor the compression ratio
  '''
  mask_palceholders = get_mask_placeholders()
  weight_placeholders = get_weight_placeholders()
  num_nonzero_row_arr = []
  num_total_row_arr = []
  num_row_size_arr = []
  num_nonzero_params = 0
  num_unique_params = 0
  num_total_params = 0

  for idx, mask_triple in enumerate(mask_palceholders):
    mask_i, mask_palceholders_i, mask_assign_op_i = mask_triple
    param_i, param_placeholder_i, param_assign_op_i = weight_placeholders[idx]
    dim_i = param_i.get_shape().as_list()

    param_val = sess.run(param_i)
    mask = sess.run(mask_i)
    param_val_masked = param_val * mask

    if np.size(dim_i) == 4:
      param_val_masked_reshaped = reshape_2D_4D(param_val_masked, target_shape=None, reshape_type=2, reshape_order='F')
      row_norm = norm(param_val_masked_reshaped, axis=1)
      num_nonzero_params += np.count_nonzero(row_norm) * np.shape(param_val_masked_reshaped)[1]
      num_unique_params += np.size(np.unique(param_val_masked_reshaped))
      num_total_params += np.prod(dim_i)
      num_nonzero_row_arr.append(np.count_nonzero(row_norm))
      num_total_row_arr.append(np.size(row_norm))
      num_row_size_arr.append(np.shape(param_val_masked_reshaped)[1])

    elif np.size(dim_i) == 2:
      row_norm = norm(param_val_masked, axis=1)
      num_nonzero_params += np.count_nonzero(row_norm) * dim_i[1]
      num_unique_params += np.size(np.unique(param_val_masked))
      num_total_params += np.prod(dim_i)
      num_nonzero_row_arr.append(np.count_nonzero(row_norm))
      num_total_row_arr.append(np.size(row_norm))
      num_row_size_arr.append(np.shape(param_val_masked)[1])
     
    # num_cluster arr only contains the cluster information for regularized layers, we need to first fill in the number of rows for unregularized layers
    if (not hps.reg_applied_layers[idx]) and (not training):
      num_cluster_arr = np.insert(num_cluster_arr, idx, np.size(row_norm))

  # calculate the list for num_param_i / num_total param
  weight_ratio_list = np.divide(np.multiply(num_total_row_arr, num_row_size_arr), float(num_total_params))

  # calculate the nonzero ratio list, nonzero ratio for each layer is defined as num_nozero_row_i/num_total_row_i
  num_total_row_arr = np.asarray(num_total_row_arr, dtype=np.float32)
  num_nonzero_row_arr = np.asarray(num_nonzero_row_arr, dtype=np.float32)
  nonzero_ratio_list = np.divide(num_nonzero_row_arr, num_total_row_arr)

  if training:
    compression_ratio_arr = np.append(np.multiply(nonzero_ratio_list[0:-1], nonzero_ratio_list[1:]), nonzero_ratio_list[-1])
    compression_ratio = np.inner(compression_ratio_arr, weight_ratio_list)
  else:
    compression_ratio_arr = np.append(np.multiply(nonzero_ratio_list[0:-1], nonzero_ratio_list[1:]), nonzero_ratio_list[-1])
    compression_ratio_arr = np.multiply(compression_ratio_arr, np.divide(num_cluster_arr, num_nonzero_row_arr))
    compression_ratio = np.inner(compression_ratio_arr, weight_ratio_list)

  print('nonzero_ratio_list: {}'.format(nonzero_ratio_list))
  print('weight_ratio_list: {}'.format(weight_ratio_list))
  print('num_nonzero_row_arr: {}'.format(num_nonzero_row_arr))
  print('num_total_row_arr: {}'.format(num_total_row_arr))
  print('num_row_size_arr: {}'.format(num_row_size_arr))

  print("At step {}, total compression ratio is: {:.4f}%".format(step, compression_ratio * 100))
  res_dict['compression_ratio_arr'].append(compression_ratio)
  np.save(FLAGS.res_dir + 'res_dict.npy', res_dict)

  return compression_ratio


def adjust_learning_rate(hps, phase, training_decay_steps, retrain_decay_steps, step):

  if phase:
    #Reset learning rate at the beginning of retraining phase
    lrn_rate = FLAGS.retraining_lr_init
    hps=hps._replace(lrn_rate=lrn_rate)
    print('Retraining!! Learning rate reset to:{0}, at step:{1}'.format(lrn_rate, step))

  elif (step in training_decay_steps) or (step in retrain_decay_steps):

    lrn_rate = hps.lrn_rate * FLAGS.lr_decay_rate
    hps=hps._replace(lrn_rate=lrn_rate)
    print('Learning rate set as:{0} at step:{1}'.format(hps.lrn_rate, step))

  return hps

def reshape_2D_4D(X, target_shape, reshape_type, reshape_order):
  """
  This function transform the 2D tensor to 4D tensor or vice versa.

  Args:
    X: input tensor, either 2D or 4D numpy array
    target_shape: target shape, must be provided if reshape_type=1
    reshape_type:
           1: 2D to 4D
           2: 4D to 2D
    reshape_order: "C" or "F", please refer to numpy.reshape

  Returns:
    Y: output tensor, either 2D or 4D numpy array
  """

  param_shape = X.shape

  #Input tensor must be either 2D or 4D
  if len(param_shape) != 2 and len(param_shape) != 4:
    raise Exception("The input tensor for the reshape_2D_4D much be either 2D or 4D!")

  #Reshape from 2D to 4D
  if reshape_type == 1:
    X_T = np.transpose(X)
    target_shape_swap = list(target_shape)
    target_shape_swap[2], target_shape_swap[3] = target_shape_swap[3], target_shape_swap[2]
    Y_swap = np.reshape(X_T, tuple(target_shape_swap), order=reshape_order)
    Y = np.swapaxes(Y_swap, 2,3)

  #Reshape from 4D to 2D
  elif reshape_type == 2:
    X_swapaxes = np.swapaxes(X, 2,3)
    param_swap_shape = X_swapaxes.shape
    Y_T = np.reshape(X_swapaxes, (np.prod(param_swap_shape[0:3]), param_swap_shape[3]), order=reshape_order)

    #Transpose the reshape matrix in order to call the row-wise prox_function
    Y = np.transpose(Y_T)

  return Y

def preprocess_hparam(batch_size, num_classes, reg_params, reg_applied_layers, param_shared_layers):
  """Crate hyperparamters tuple and results dictionary for running the code.
    Returns:
      hps: hyperparameters
      res_dict: results dictionary
  """
  if not os.path.exists(FLAGS.plot_dir):
    os.makedirs(FLAGS.plot_dir)
  if not os.path.exists(FLAGS.res_dir):
    os.makedirs(FLAGS.res_dir)
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)
  if not os.path.exists(FLAGS.retrain_dir):
    os.makedirs(FLAGS.retrain_dir)
  if not os.path.exists(FLAGS.eval_dir):
    os.makedirs(FLAGS.eval_dir)

  #Initialize hyperparameters
  hps = HParams(batch_size=batch_size,
                num_classes=num_classes,
                min_lrn_rate=0.0001,
                lrn_rate=FLAGS.training_lr_init,
                use_weight_decay=FLAGS.use_weight_decay,
                weight_decay_rate=FLAGS.weight_decay_rate,
                relu_leakiness=0.1,
                optimizer='mom',
                reg_params=reg_params,
                reg_applied_layers=reg_applied_layers,
                param_shared_layers=param_shared_layers)

  #Create dictionary for storing the intermediate results
  np.save(FLAGS.log_root + 'hps.npy', hps._asdict())
  res_dict = {}
  res_dict['compression_ratio_arr'] = []
  res_dict['num_cluster_arr'] = []
  res_dict['num_nonzero_row_arr'] = []
  res_dict['test_accur_arr'] = []
  res_dict['test_loss_arr'] = []
  res_dict['training_accur_arr'] = []
  res_dict['steps'] = []

  return hps, res_dict


def set_param_share(data_dir, fileName, param_shared_layers):

  with open(data_dir + fileName) as f:
    res = f.read()
    numbers = re.findall(r'\d+', res)
    clusters = numbers[-1:-39:-3]
    clusters = np.array([i for i in clusters[-1:-14:-1]]).astype('float32')
    nonzeros = numbers[-2:-39:-3]
    nonzeros = np.array([i for i in nonzeros[-1:-14:-1]]).astype('float32')
    
    nonzeros.astype('float')
    sharing = np.divide(clusters, nonzeros)
    False_idx = np.where(sharing>0.95)[0] + 1
    
    param_shared_layers = np.asarray(param_shared_layers)
    param_shared_layers[False_idx] = False

    print('sharing:{}, param_shared_layers:{}'.format(sharing, param_shared_layers))

  return param_shared_layers
