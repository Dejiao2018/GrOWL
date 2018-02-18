# -*- coding: utf-8 -*-
'''
This modules contains functions necessary for applying group OWL
to the parameters
    1. reshape_2D_4D
    2. reg_params_init
    3. apply_growl
    4. apply_owl_prox
    5. update_mask
'''

from __future__ import division, print_function, absolute_import
import sys
sys.path.append('./owl_projection')

import tensorflow as tf
import numpy as np
import utils_nn
from projectedOWL import proxOWL
from numpy.linalg import norm
from utils_retrain import group_averaging
from utils_general import reshape_2D_4D
import matplotlib.pyplot as plt
from utils_plot import hist_plot


def reg_params_init(sess, config):
    '''
    This function initializes the regularization paramters.

    Args:
        sess: the predefined computation graph.
        config: the yaml configuration file.

    Returns:
        layer_owl_params: n-tuple, each elements is an array containing the weights
                          of the corresponding layer.
    '''

    weight_placeholder = utils_nn.get_weight_placeholders()
    layer_owl_params = []
    min_num_row = float("Inf")
    if config['PLD_transition'] == 0:
        # read out the minimum number of rows
        for idx, triple in enumerate(weight_placeholder):
            param_i, placeholder_i, assign_op_i = triple
            param_shape = sess.run(tf.shape(param_i))
            if param_shape[0] < min_num_row:
                min_num_row = param_shape[0]


    # iterates through all layers, idx is the layer number
    for idx, triple in enumerate(weight_placeholder):

        param_i, placeholder_i, assign_op_i = triple

        # OWL weights should be applied to the columns of the weight matrix
        param_shape = sess.run(tf.shape(param_i))

        if config['use_owl']:
            reg_params = config['owl_params']
        else:
            reg_params = config['growl_params']

        # for fully connected layers
        if np.size(param_i.get_shape().as_list()) == 2:

            lambda_1 = np.float32(reg_params[idx][0])
            lambda_2 = np.float32(reg_params[idx][1])
            if (lambda_1 < 0) | (lambda_2 < 0):
                raise Exception('regularization parameters must be non-negative')

            #OSCAR type parameters: lambda_i = lambda_1 + (n - i) * lambda_2
            if config['reg_params_type'] == 'OSCAR':
                param_index = np.linspace(start=param_shape[0]-1, stop=0, num=param_shape[0])
                layer_owl_params.append(lambda_1 + lambda_2 * param_index)

            elif config['reg_params_type'] == 'PLD': # partiial linear decreasing
                if config['PLD_transition'] != 0:
                    transition_ind = np.floor(param_shape[0]*config['PLD_transition']) -1
                else:
                    transition_ind = min_num_row
                param_index = np.linspace(start=transition_ind-1, stop=0, num=transition_ind)
                param_index = np.append(param_index, np.zeros([1, int(param_shape[0]-transition_ind)]))
                layer_owl_params.append(lambda_1 + lambda_2 * param_index)
                print(np.max(lambda_1 + lambda_2 * param_index))

        # for convolutional layers
        elif np.size(param_i.get_shape().as_list()) == 4:
            #OSCAR type parameters: lambda_i = lambda_1 + (n - i) * lambda_2

            lambda_1 = np.float32(reg_params[idx][0])
            lambda_2 = np.float32(reg_params[idx][1])
            if (lambda_1 < 0) | (lambda_2 < 0):
                raise Exception('regularization parameters must be non-negative')

            # calculate the number of rows according to the regularization type
            if config['conv_reg_type'] == 1:
                row_num = param_shape[3]
            elif config['conv_reg_type'] == 2:
                row_num = param_shape[2]
            elif config['conv_reg_type'] == 3:
                row_num = np.prod(param_shape[2:])
            elif config['conv_reg_type'] == 4:
                row_num = row_num = np.prod(param_shape[:3])
            else:
                raise Exception('Please specify the regularization type for convolutional layers')

            if config['reg_params_type'] == 'OSCAR':
                param_index = np.linspace(start=row_num-1, stop=0, num=row_num)
                layer_owl_params.append(lambda_1 + lambda_2 * param_index)
            elif config['reg_params_type'] == 'PLD':
                if config['PLD_transition'] != 0:
                    transition_ind = np.floor(param_shape[0]*config['PLD_transition']) -1
                else:
                    transition_ind = min_num_row
                param_index = np.linspace(start=row_num-1, stop=0, num=row_num)
                param_index = np.append(param_index, np.zeros([1, param_shape[0]-transition_ind]))
                layer_owl_params.append(lambda_1 + lambda_2 * param_index)

    assert len(layer_owl_params) == len(weight_placeholder)

    return layer_owl_params

def apply_group_lasso(W, weights):

    W_norm = norm(W, axis=1)

    # apply prox op of Lasso to norms of rows
    new_W_norm = np.maximum(W_norm - weights[0], 0)

    # compute group owl
    new_W = np.zeros_like(W)

    for i in range(W.shape[0]):

        # print('w_norm: {}'.format(W_norm[i]))
        # print('new_w_norm: {}'.format(new_W_norm[i]))

        if W_norm[i] < np.finfo(np.float32).eps:
            new_W[i,:] = 0 * W[i,:]
        else:
            new_W[i,:] = new_W_norm[i] * W[i,:] / W_norm[i]

    return new_W


def apply_growl(W, weights):


    W_norm = norm(W, axis=1)

    # apply prox op to norms of rows
    new_W_norm=proxOWL(W_norm, weights)
    # print('update proximal {}'.format(np.sum(np.abs(new_W_norm - W_norm))))

    # compute group owl
    new_W = np.zeros_like(W)

    for i in range(W.shape[0]):

        if W_norm[i] < np.finfo(np.float32).eps:
            new_W[i,:] = 0 * W[i,:]
        else:
            new_W[i,:] = new_W_norm[i] * W[i,:] / W_norm[i]

    return new_W

def apply_owl_prox(sess, learning_rate, layer_reg_params, config):
    '''
    Updates the weights parameter of each layer

    Args:
        sess: the comptutaion graph
        learning_rate: the predefined learning rate
        layer_reg_params: owl parameters, initially created by reg_params_init
        config: yaml configuration file

    Returns:
        None
    '''

    # get weights of the network
    weight_placeholders = utils_nn.get_weight_placeholders()
    learning_rate_val = sess.run(learning_rate)

    for idx, triple in enumerate(weight_placeholders):

        #Don't apply owl/growl if told not to
        if not config['owl_applied_layers'][idx]:
            continue

        param_i, placeholder_i, assign_op_i = triple
        param_val = sess.run(param_i)
        dim_i = np.size(param_val.shape)

        if dim_i == 2:
            # if the weight tensor is 2 dimensional, then it's fc layer
            if config['use_growl']:
                prox_param_val = apply_growl(param_val, learning_rate_val * layer_reg_params[idx])
            else:
                prox_param_val = apply_group_lasso(param_val, learning_rate_val * layer_reg_params[idx])

        elif dim_i == 4:
            # If the weight tensor is 4-dimensional, then it's the convolutional layer
            # For convolutional layer, we need to first reshape 4D tensor to 2D matrix
            reduced_param_val = reshape_2D_4D(param_val, target_shape=None, reshape_type=2,
                                              reshape_order='F', config=config)
            if config['use_growl']:
                reduced_prox_param_val = apply_growl(reduced_param_val, learning_rate_val * layer_reg_params[idx] * np.sqrt(np.shape(reduced_param_val)[1]))
            else:
                reduced_prox_param_val = apply_group_lasso(reduced_param_val, learning_rate_val * layer_reg_params[idx] * np.sqrt(np.shape(reduced_param_val)[1]))

            #Now reshape the 2D matrix back to 4D tensor
            prox_param_val = reshape_2D_4D(reduced_prox_param_val, target_shape=param_val.shape,
                                           reshape_type=1, reshape_order='F', config=config)

        # assign the new weights to param_i using the assign_op_i
        # refer to utils_nn.py for details of assign_op_i
        sess.run(assign_op_i, feed_dict={placeholder_i:prox_param_val})


def update_mask(sess, epoch, learning_rate, threshold, phase, config, group_info=None, get_nonzero_idx_flag=False):
    '''
    update the mask

    Args:
        sess: the computation graph
        learning_rate: the predefined learning rate
        threshold: the pruning threshold, this may help avoid the floating number error
                   occured during the masking process
        phase: False for training, True for retraining. If Ture, then enforce parameter sharing
        config: the yaml configuration file
        group_info: the group information. A list of tuples, each tuple contains the index of the rows
        which belongs to the same group

    Returns:
        compression_ratio: percentage, the ratio between nonzero paramters and total parameters
    '''

    mask_palceholders = utils_nn.get_mask_placeholders()
    weight_placeholders = utils_nn.get_weight_placeholders()
    num_total_params = 0
    num_nonzero_params = 0
    num_unique_params = 0
    compression_ratio = 1

    #count the zero valued layers in order to avoiding the nonsense results
    num_zero_layers = 0

    assert len(mask_palceholders) == len(weight_placeholders)

    # track the index of the regularized layer, for retrieving the group info
    idx_true_layer = 0

    for idx, mask_triple in enumerate(mask_palceholders):

        #Don't apply owl/growl if told not to
        if not config['owl_applied_layers'][idx]:
            continue

        mask_i, mask_palceholders_i, mask_assign_op_i = mask_triple
        param_i, param_placeholder_i, param_assign_op_i = weight_placeholders[idx]
        dim_i = param_i.get_shape().as_list()

        # recover the masked weights to zeros if they drifted
        param_val = sess.run(param_i)
        mask = sess.run(mask_i)
        param_val_masked = param_val * mask

        learning_rate_val = sess.run(learning_rate)

        # if apply to convolutional layer, compute the reshaped matrix
        if np.size(dim_i) == 4:

            # first reshape the 4 dimensional tensor into a matrix
            param_val_masked_reshaped = reshape_2D_4D(param_val_masked, target_shape=None, reshape_type=2,
                                              reshape_order='F', config=config)
            # param_val_masked_reshaped = np.reshape(param_val_masked, (dim_i[3], np.prod(dim_i[0:3])))

            # This is the pruning process
            row_norm = norm(param_val_masked_reshaped, axis=1)
            print('min row norm {:.4f}'.format(np.min(row_norm)))
            print('current epoch {}'.format(epoch+1))
            if epoch==0 or (epoch+1) % config['row_norm_freq'] == 0:
                hist_plot(idx, epoch, phase, row_norm[row_norm>0], config)

            zero_row_idx = np.where(row_norm <=threshold)
            print('masked neurons: {}; total neurons: {}'.format(np.size(zero_row_idx), np.size(row_norm)))
            param_val_masked_reshaped[zero_row_idx[0], :] = 0

            # in retraining process, enforce parameter sharing
            if phase:
                param_val_masked_reshaped = group_averaging(param_val_masked_reshaped, group_info[idx_true_layer])

            #Now reshape the 2D matrix back to 4D tensor
            param_val_masked = reshape_2D_4D(param_val_masked_reshaped, target_shape=tuple(dim_i),
                                           reshape_type=1, reshape_order='F', config=config)

            # update parameter
            sess.run(param_assign_op_i, feed_dict={param_placeholder_i:param_val_masked})

            # update the mask in training process
            # Only update mask at the locations that corresponding to zero valued rows
            if not phase:
                #4D to 2D
                mask_reshaped = reshape_2D_4D(mask, target_shape=None, reshape_type=2,
                                              reshape_order='F', config=config)
                #prune redundant rows
                mask_reshaped[zero_row_idx[0], :] = 0
                #back to 4D
                mask = reshape_2D_4D(mask_reshaped, target_shape=tuple(dim_i),
                                           reshape_type=1, reshape_order='F', config=config)

                sess.run(mask_assign_op_i, feed_dict={mask_palceholders_i:mask})

        # for fully connected layer
        elif np.size(dim_i) == 2:
            if config['use_growl'] | config['use_group_lasso']:

                # This is the pruning process
                row_norm = norm(param_val_masked, axis=1)
                print('min row norm {:.4f}'.format(np.min(row_norm)))
                print('current epoch {}'.format(epoch+1))
                if epoch==0 or (epoch+1) % config['row_norm_freq'] == 0:
                    hist_plot(idx, epoch, phase, row_norm[row_norm>0], config)

                zero_row_idx = np.where(row_norm <=threshold)

                nonzero_row_idx = np.where(row_norm > threshold)
                np.save(config['plot_dir']+'nonzero_row_idx.npy', nonzero_row_idx)

                print('masked rows: {}; total rows: {}'.format(np.size(zero_row_idx), np.size(row_norm)))
                param_val_masked[zero_row_idx[0], :] = 0

                # in retraining process, enforce parameter sharing, do not update the mask
                if phase:
                    param_val_masked = group_averaging(param_val_masked, group_info[idx_true_layer])

                # update parameter
                sess.run(param_assign_op_i, feed_dict={param_placeholder_i:param_val_masked})

                # update the mask in training process
                # Only update mask at the locations that corresponding to zero valued rows
                if not phase:
                    mask[zero_row_idx[0], :] = 0
                    sess.run(mask_assign_op_i, feed_dict={mask_palceholders_i:mask})

        layer_nonzero_params = np.count_nonzero(param_val_masked)
        print("update mask of layer: {0}, total:{1}, nonzeros:{2}, uniqs:{3}".format(idx,
            np.size(param_val_masked),
            layer_nonzero_params,
            len(np.unique(param_val_masked))))

        num_total_params = num_total_params + np.size(param_val_masked)
        print("num_total_params:{0}, param_val_size:{1}".format(num_total_params, np.size(param_val_masked)))
        num_nonzero_params = num_nonzero_params + layer_nonzero_params
        num_unique_params = num_unique_params + len(np.unique(param_val_masked))
        idx_true_layer = idx_true_layer + 1

        #record the zero valued layers
        if np.size(row_norm) - np.size(zero_row_idx[0]) <= 3:
            num_zero_layers += 1


    # in training, we care about nonzero parameters
    if not phase:
        compression_ratio = num_nonzero_params/num_total_params
    # in retraining, we care about unique parameters
    else:
        compression_ratio = num_unique_params/num_total_params
    # compression_raito = 0

    print("Total compression ratio is: {:.4f}%".format(compression_ratio * 100))

    return compression_ratio, num_zero_layers
