'''
This modules contains functions necessary for retraining the model after the
initial training with OWL or group OWL

  1. get_group_info_owl
  2. retrain_owl
  3. get_group_info_growl
  4. retrain_group_owl
'''
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from sklearn.cluster import AffinityPropagation
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from utils_GrOWL import reshape_2D_4D
from flags import FLAGS
import matplotlib.pyplot as plt
from utils_nn import get_mask_placeholders, get_weight_placeholders


def display_similarity(sess, step, hps, res_dict):
  '''
  display the pairwise similarity of the rows of the weight matrix.

  Args:
    sess: the computation graph
    step: current step
    hps: hyperparameters
    res_dict: results dictionary

  Return:
  '''
  num_nonzero_rows_tuple, num_clusters_tuple = tuple(), tuple()
  mask_placeholders = get_mask_placeholders()
  weight_placeholders = get_weight_placeholders()

  assert len(mask_placeholders) == len(weight_placeholders)

  group_info = []
  num_clusters_arr = []
  threshold = np.finfo(np.float32).eps

  #Track the nonzero row index
  nonzero_row_index = []

  # only process the layers we applied grOWL on
  for idx,  mask_triple in enumerate(mask_placeholders):

    # double check we have turned on grOWL for the layer
    if not hps.reg_applied_layers[idx]:
      continue

    mask_i, mask_placeholders_i, mask_assign_op_i = mask_placeholders[idx]
    param_i, param_placeholder_i, param_assign_op_i = weight_placeholders[idx]
    dim_i = param_i.get_shape().as_list()

    param_masked = tf.multiply(mask_i, param_i)
    if np.size(dim_i) == 2:
      param_masked_val = sess.run(param_masked)
      print("layer:{0}, param_masked_val.shape:{1}".format(idx, param_masked_val.shape))
    elif np.size(dim_i) == 4:
      param_masked_4D = sess.run(param_masked)
      param_masked_val = reshape_2D_4D(param_masked_4D, target_shape=None,
                       reshape_type=2, reshape_order='F')
      print("layer:{0}, param_masked_val.shape:{1}".format(idx, param_masked_val.shape))

  
    # first retrieve nonzero rows from the parameter matrix
    row_norm = norm(param_masked_val, axis=1)
    row_norm[row_norm < threshold] = 0 # remove very small row norms
    num_nonzero_rows = np.count_nonzero(row_norm)
    nonzero_row_idx = np.flatnonzero(row_norm)
    nonzero_rows = param_masked_val[row_norm>0, :]
    norm_nonzero_rows = norm(nonzero_rows, axis=1)

    #calculate the display similarity matrix without removing the zero valued rows
    num_rows=row_norm.size
    display_similarity_val = np.zeros([num_rows, num_rows])
    display_similarity_val_partial = np.zeros([nonzero_row_idx.size, nonzero_row_idx.size])

    zero_row_idx = np.where(row_norm<threshold)[0]
    for k in zero_row_idx:
      display_similarity_val[k,:] = 0
      display_similarity_val[:, k] = 0

    partial_idx = np.arange(nonzero_row_idx.size)

    for idx_newi, i in np.nditer([partial_idx,nonzero_row_idx]):
      for idx_newj, j in np.nditer([partial_idx,nonzero_row_idx]):
        if row_norm[i] > row_norm[j]:
          display_similarity_val[i,j] = display_similarity_val_partial[idx_newi, idx_newj] = np.dot(param_masked_val[i,:], param_masked_val[j,:])/row_norm[i]**2
        else:
          display_similarity_val[i,j] = display_similarity_val_partial[idx_newi, idx_newj] = np.dot(param_masked_val[i,:], param_masked_val[j,:])/row_norm[j]**2

    #Save the similarity matrix
    np.save(FLAGS.res_dir + 'similarity_{}.npy'.format(int(step/390)), display_similarity_val)

    # Cluster the paramter matrix with affinity propagation
    if num_nonzero_rows > 1:

      preference_val = FLAGS.preference
      print('CLUSTERING ROWS WITH PREFERENCE BVALUE:{}'.format(preference_val))
      af = AffinityPropagation(affinity='precomputed', preference=preference_val).fit(display_similarity_val_partial)

      cluster_centers_indices = af.cluster_centers_indices_
      num_clusters = np.size(cluster_centers_indices)
      with open(FLAGS.res_dir + 'cluster.txt', 'a') as f:
        print('  idx: {}, Nonzero rows: {}, Number of Clusters: {}'.format(idx, num_nonzero_rows, num_clusters))
        f.write('  idx: {}, Nonzero rows: {}, Number of Clusters: {}\n'.format(idx, num_nonzero_rows, num_clusters))
      num_clusters_arr.append(num_clusters)


      # get the labels for the nonzero rows and construct the tuple list for storing the group
      # information
      labels = af.labels_
      group_info_i = [tuple()] * num_clusters
      for i in range(len(labels)):
        group_info_i[labels[i]] = group_info_i[labels[i]] + (nonzero_row_idx[i], )

      group_info.append(group_info_i)
      # put the clustering results in the tuple for return
      num_nonzero_rows_tuple = num_nonzero_rows_tuple + (num_nonzero_rows, )
      num_clusters_tuple = num_clusters_tuple + (num_clusters, )

      nonzero_row_index.append(nonzero_row_idx)

  # store the intermediate results
  res_dict['num_cluster_arr'].append(num_clusters_tuple)
  res_dict['num_nonzero_row_arr'].append(num_nonzero_rows_tuple)
  np.save(FLAGS.res_dir + 'res_dict.npy', res_dict)
  np.save(FLAGS.res_dir + 'nonzero_index.npy', nonzero_row_index)
  np.save(FLAGS.train_dir + 'group_info_{}.npy'.format(int(step/390)), group_info)


  return group_info, num_clusters_arr


def group_averaging(param_val_masked, group_info):
  '''
  This function averages the rows which belongs to the same group

  Args:
    param_val_masked: the masked parameter matrix
    group_info: list of tuples, each tuple contains the rows indices that belong to the same group

  Return:
    param_val_avg: the averaged parameter rows after enforcing parameter sharing
  '''
  param_val_avg = param_val_masked

  for group_i in group_info:

    group_i = list(group_i)
    avg_param_i = np.average(param_val_masked[group_i, :], axis=0)
    param_val_avg[group_i, :] = avg_param_i

  return param_val_avg


def apply_param_share(sess, group_info, hps):
  """Parameter sharing for the retraining phase
  Args:
    sess: the computation graph
    group_info: the group information. A list of tuples, each tuple contains the index of the rows
    which belongs to the same group
    hps: 
  """

  weight_placeholders = get_weight_placeholders()
  # Track the index of the regularized layer, for retrieving the group info
  idx_true_layer = 0
  for idx, triple in enumerate(weight_placeholders):
    #Don't apply param share to the non_regularizer applied layers
    if not hps.reg_applied_layers[idx]: 
      continue
    
    #Only apply param share to those layers that from the pattern of clusters
    if not hps.param_shared_layers[idx]:
      #Update group idx
      idx_true_layer = idx_true_layer + 1
      continue

    param_i, param_placeholder_i, param_assign_op_i = triple
    dim_i = param_i.get_shape().as_list()
    param_val = sess.run(param_i)

    if np.size(dim_i) == 4:

      #reshape the 4D tensor to a 2D matrix
      param_val_reshaped = reshape_2D_4D(param_val, target_shape=None, reshape_type=2,
                        reshape_order='F')
      
      #retrain with parameter sharing
      param_val_reshaped_shared = group_averaging(param_val_reshaped, group_info[idx_true_layer])

      #back to 4D tensor
      param_val_shared = reshape_2D_4D(param_val_reshaped_shared, target_shape=tuple(dim_i),
                       reshape_type=1, reshape_order='F')

    elif np.size(dim_i) == 2:
     
      param_val_shared = group_averaging(param_val, group_info[idx_true_layer])

    #Update parameter
    sess.run(param_assign_op_i, feed_dict={param_placeholder_i:param_val_shared})

    #Update group idx
    idx_true_layer = idx_true_layer + 1


