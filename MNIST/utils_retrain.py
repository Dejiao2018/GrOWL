'''
This modules contains functions necessary for retraining the model after the
initial training with OWL or group OWL

    1. get_group_info_owl
    2. retrain_owl
    3. get_group_info_growl
    4. retrain_group_owl
'''
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('./growl_param_share') #TODO
sys.path.append('./owl_param_share') #TODO

import tensorflow as tf
import numpy as np
from sklearn.cluster import AffinityPropagation
import utils_nn
from numpy.linalg import norm
from sklearn.preprocessing import normalize
from utils_plot import imagesc
from utils_general import reshape_2D_4D
import matplotlib.pyplot as plt
import yaml


def display_similarity(sess, epoch, get_group, config):

    '''
    display the pairwise similarity of the rows of the weight matrix. and plot the
    clustering results.

    Args:
        sess: the computation graph
        epoch: current epoch

    Return:
        num_nonzero_rows_tuple:
        num_clusters_tuple:
        group_info:

    '''

    num_nonzero_rows_tuple, num_clusters_tuple = tuple(), tuple()

    mask_placeholders = utils_nn.get_mask_placeholders()
    weight_placeholders = utils_nn.get_weight_placeholders()

    num_layers = 0

    threshold = np.finfo(np.float32).eps

    assert len(mask_placeholders) == len(weight_placeholders)
    group_info = []

    # only process the layers we applied grOWL on
    for idx,  mask_triple in enumerate(mask_placeholders):

        # double check we have turned on grOWL for the layer
        if not config['owl_applied_layers'][idx]:
            continue

        num_layers = num_layers + 1
        mask_i, mask_placeholders_i, mask_assign_op_i = mask_placeholders[idx]
        param_i, param_placeholder_i, param_assign_op_i = weight_placeholders[idx]
        dim_i = param_i.get_shape().as_list()

        # recover the masked weights to zeros if they drifted
        param_masked = tf.multiply(mask_i, param_i)

        if np.size(dim_i) == 2:
            param_masked_val = sess.run(param_masked)
        elif np.size(dim_i) == 4:
            param_masked_4D = sess.run(param_masked)
            param_masked_val = reshape_2D_4D(param_masked_4D, target_shape=None,
                                           reshape_type=2, reshape_order='F', config=config)

        if config['similarity'] == 'cosine':
            #normalize each row
            param_normalize_val_raw =  normalize(param_masked_val, axis=1)
            param_normalize_val = np.maximum(param_normalize_val_raw, 1e-12)
            row_norm = norm(param_normalize_val, axis=1)
            num_nonzero_rows = np.count_nonzero(row_norm)
            nonzero_row_idx = np.flatnonzero(row_norm)

            # cosine similarity
            similarity_val = np.dot(param_normalize_val, param_normalize_val.T)

            #In this case, display_similarity_val equals to similarity_val
            display_similarity_val = similarity_val

        elif config['similarity'] == 'norm_euclidean':
            # first retrieve nonzero rows from the parameter matrix
            row_norm = norm(param_masked_val, axis=1)
            row_norm[row_norm < threshold] = 0 # remove very small row norms
            num_nonzero_rows = np.count_nonzero(row_norm)
            nonzero_row_idx = np.flatnonzero(row_norm)
            nonzero_rows = param_masked_val[row_norm>0, :]
            norm_nonzero_rows = norm(nonzero_rows, axis=1)
            # then compute the normalized Euclidean norm similarity matrix, we should take the negative
            # so that similar rows have a larger affinity value
            similarity_val = np.zeros([num_nonzero_rows, num_nonzero_rows])
            for i in range(num_nonzero_rows):
                for j in range(num_nonzero_rows):
                    if norm_nonzero_rows[i] > norm_nonzero_rows[j]:
                        similarity_val[i,j] = -norm(nonzero_rows[i,:] - nonzero_rows[j,:])/norm_nonzero_rows[j]
                    else:
                        similarity_val[i,j] = -norm(nonzero_rows[i,:] - nonzero_rows[j,:])/norm_nonzero_rows[i]

            #calculate the display similarity matrix without removing the zero valued rows
            num_rows = row_norm.size
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

        # save the similarity val
        if get_group:
            sim_name = config['plot_dir'] + 'epoch_{0}_similarity_{1}.npy'.format(epoch, idx+1)
            np.save(sim_name, display_similarity_val)

        #Visualize the similarity matrix
        num_rows = display_similarity_val.shape[0]
        row_idx = np.linspace(1, num_rows, float(num_rows))

        x, y = np.meshgrid(row_idx, row_idx)

        # plot the grid search results for accuracy
        figName = config['plot_dir'] + 'epoch_{0}_similarity_{1}.png'.format(epoch, idx+1)
        if config['similarity'] == 'norm_euclidean':
            imagesc(x, y, display_similarity_val, 'row_idx', 'row_idx', 'similarity matrix of layer {0}'.format(idx+1),
             False, figName)

        similarity_val_arr = np.reshape(similarity_val, [-1])
        display_similarity_val_arr = np.reshape(display_similarity_val, [-1])

        # save the nonzero similarity value
        nonzero_display_similarity_val_arr = display_similarity_val_arr[np.abs(display_similarity_val_arr)>0]
        print('Layer {}, the median of the similarity_val is {}'.format(idx, np.median(nonzero_display_similarity_val_arr)))
        if get_group:
            np.savetxt(config['plot_dir'] + 'sim_vec_epoch{}.csv'.format(epoch), display_similarity_val_arr)
        plt.hist(nonzero_display_similarity_val_arr)
        plt.title('Similarity histogram at epoch {}'.format(epoch+1))
        plt.savefig(config['plot_dir'] + 'layer{}_train_sim_{}.jpg'.format(idx, epoch))
        plt.close()

        # construct the nonzero display_similarity array for affinity propagation
        # print('nonzero similarity {}'.format(nonzero_display_similarity_val_arr.size))
        # nonzero_sim_matsize = np.sqrt(nonzero_display_similarity_val_arr.size)
        # assert np.sqrt(nonzero_display_similarity_val_arr.size) == nonzero_row_idx.size, 'the nonzero numbers obtained from similiary values does not match true nonzero rows'
        # nonzero_display_similarity = np.reshape(nonzero_display_similarity_val_arr, [nonzero_sim_matsize, nonzero_sim_matsize])



        # # zip the original row indices with new indices range(nonzero_sim_matsize)
        # row_sim_dict = list(zip(nonzero_row_idx, range(len(nonzero_row_idx))))


        # Cluster the paramter matrix with affinity propagation
        if num_nonzero_rows > 0:
            if config['similarity'] == 'euclidean':
                af = AffinityPropagation().fit(nonzero_param_val)
            elif config['similarity'] == 'norm_euclidean':
                # af = AffinityPropagation(affinity='precomputed').fit(similarity_val)
                af = AffinityPropagation(affinity='precomputed', preference=config['preference']).fit(display_similarity_val_partial)

            cluster_centers_indices = af.cluster_centers_indices_
            num_clusters = np.size(cluster_centers_indices)

            print("number of clusters {}".format(num_clusters))

            # get the labels for the nonzero rows and construct the tuple list for storing the group information. group_info_i is only for the ith layer
            if get_group:
                labels = af.labels_
                group_info_i = [tuple()] * num_clusters
                for i in range(len(labels)):
                    group_info_i[labels[i]] = group_info_i[labels[i]] + (nonzero_row_idx[i], )

                group_info.append(group_info_i)

            # put the clustering results in the tuple for return
            num_nonzero_rows_tuple = num_nonzero_rows_tuple + (num_nonzero_rows, )
            num_clusters_tuple = num_clusters_tuple + (num_clusters, )

            print('Nonzero rows: {}, Number of Clusters: {}'.format(num_nonzero_rows, num_clusters))
        else:
            num_nonzero_rows_tuple = num_nonzero_rows_tuple + (0, )
            num_clusters_tuple = num_clusters_tuple + (0, )

            print('Nonzero rows: {}, Number of Clusters: {}'.format(0, 0))



    return num_nonzero_rows_tuple, num_clusters_tuple, group_info



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
