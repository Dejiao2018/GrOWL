'''
This is the main module, which is composed of two parts:
part 1: network architecture
part 2: training

This code uses single hidden layer, serves for prototyping
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from networks import LeNet_single
from utils_OWL import reg_params_init, apply_owl_prox, update_mask
from utils_plot import imagesc, cluster_plot, accuracy_compression_plot
from utils_retrain import display_similarity

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist_data = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)

gpu = True # if set to True, the variables will be initialized on GPU
mask_threshold = np.finfo(np.float32).eps # the mask threshold for the row norm

def train_mnist(config):

    num_cluster_arr = []
    num_nonzero_row_arr = []
    test_accur_arr = []
    compression_ratio_arr = []

    with tf.Graph().as_default():


        X_image_flat = tf.placeholder(tf.float32, [None, 28*28])
        Y_true = tf.placeholder(tf.float32, [None, 10])
        phase = tf.placeholder(tf.bool, name='phase') # for batch normalization

        #Call the inference network
        logits = LeNet_single(X_image_flat, phase, gpu, config)

        # softmax cross entropy loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=logits, name='cross_entropy'), name='cross_entropy_mean')
        tf.add_to_collection('losses', cross_entropy)
        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # prediction accuracy
        correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y_true,1))
        pred_accur = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('training_accuracy', pred_accur)

        #########################################
        # Variables for training
        #########################################
        # Schedule the learning rate
        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = int(mnist_data.train.num_examples/config['batch_size'] * config['num_epochs_per_decay'])
        lr = tf.train.exponential_decay(config['learning_rate'],
            global_step,
            decay_steps,
            config['learning_rate_decay_factor'],
            staircase=True)
        tf.summary.scalar('learning_rate', lr)

        # Control dependency, refer to the doc of tf.layers.batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before applying gradients
            # opt = tf.train.GradientDescentOptimizer(lr)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            grads_and_vars = opt.compute_gradients(loss)
            optimizer = opt.apply_gradients(grads_and_vars, global_step=global_step)

        # Add histograms for gradients
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        #########################################
        # Variables for retraining
        #########################################
        # create the global step and optimizer for retrain process
        global_step_retrain = tf.Variable(0, name='global_step_retrain', trainable=False)

        lr_retrain = tf.train.exponential_decay(config['learning_rate'],
            global_step_retrain,
            decay_steps,
            config['learning_rate_decay_factor'],
            staircase=True)
        tf.summary.scalar('learning_rate', lr_retrain)

        # Control dependency, refer to the doc of tf.layers.batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        # Ensures that we execute the update_ops before applying gradients
            # opt = tf.train.GradientDescentOptimizer(lr)
            opt_retrain = tf.train.MomentumOptimizer(lr_retrain, 0.9)
            grads_and_vars_retrain = opt.compute_gradients(loss)
            optimizer_retrain = opt.apply_gradients(grads_and_vars_retrain, global_step=global_step_retrain)

        # Summary op
        summary_op = tf.summary.merge_all()

        # Initializer for the variables
        init = tf.global_variables_initializer()

        # Saver op
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(config['summary_dir'], graph=tf.get_default_graph())


        # Launch the graph
        with tf.Session() as sess:

            sess.run(init)

            if config['use_growl'] | config['use_group_lasso']:
                layer_owl_params = reg_params_init(sess, config)


            #####################################################################
            # TRAINING
            #####################################################################
            batch_per_epoch = int(mnist_data.train.num_examples/config['batch_size'])
            get_nonzero_idx_flag = True
            test_accur = 0.1
            compression_ratio = 1
            for epoch in range(config['num_epochs']):
                for i in range(batch_per_epoch):
                    inputs, labels = mnist_data.train.next_batch(config['batch_size'])
                    # Run the optimizer, record the parameters if global_step % freq == 0
                    if i % config['tensorboard_freq'] == 0:

                        summary_extend = tf.Summary()

                        _, summary, train_loss = sess.run([optimizer, summary_op, loss],
                            feed_dict={X_image_flat: inputs,
                            Y_true: labels,
                            phase: True})

                        lr_t = sess.run(opt._learning_rate)

                        #Add compression ratio and testing accuracy to tensorboard
                        summary_extend.ParseFromString(summary)
                        summary_extend.value.add(tag='testing_accuracy', simple_value=test_accur)
                        summary_extend.value.add(tag='compression_ratio', simple_value=compression_ratio)

                        step = sess.run(global_step)
                        writer.add_summary(summary_extend, step)

                    else:
                        _, train_loss = sess.run([optimizer, loss],
                            feed_dict={X_image_flat: inputs,
                            Y_true: labels,
                            phase: True})

                    if config['prox_update_iter'] & (config['use_growl'] | config['use_group_lasso']):
                        # apply OWL to weights every iteration
                        apply_owl_prox(sess, lr, layer_owl_params, config)


                if (not config['prox_update_iter']) & (config['use_growl'] | config['use_group_lasso']):
                    # apply OWL to weights every epoch
                    apply_owl_prox(sess, lr, layer_owl_params, config)


                # update the mask every epoch
                if (config['use_growl'] | config['use_group_lasso']) & (epoch % config['mask_update_freq'] == 0):

                    # update the mask and get the compression ratio
                    compression_ratio, zero_layers = update_mask(sess, epoch, lr, mask_threshold, False, config, False)
                    print('compression ratio is {}'.format(compression_ratio))
                    #Early stop there exists zero value layers
                    if zero_layers >=1:
                        print("There exists zero value layer, RETURN")
                        return test_accur, compression_ratio

                # testing
                test_accur= sess.run(pred_accur,
                    feed_dict={X_image_flat: mnist_data.test.images,
                    Y_true: mnist_data.test.labels,
                    phase: False})

                # store the testing accuracy and compression ratio
                test_accur_arr.append(test_accur)
                compression_ratio_arr.append(compression_ratio)

                # record results
                if (epoch+1 == 1) or ((epoch+1) % config['display_similarity'] == 0):

                    # saver.save(sess, config['summary_dir']+'train_model/', global_step=step)
                    summary_dir = config['summary_dir'] + 'train_model/'
                    if not os.path.exists(summary_dir):
                        os.makedirs(summary_dir)
                    saver.save(sess, summary_dir, global_step=step)

                    num_nonzero_row_tuple, num_cluster_tuple, _ = display_similarity(sess, epoch+1, False, config)
                    # store the returned tuple in the array
                    num_cluster_arr.append(num_cluster_tuple)
                    num_nonzero_row_arr.append(num_nonzero_row_tuple)


                print('Epoch:{}, lr={}, train_loss={:.4f}, accuracy={:.4f}'.format(epoch+1, lr_t, train_loss, test_accur))

            # convert the array into np.array for further operation
            num_cluster_arr = np.array(num_cluster_arr)
            num_nonzero_row_arr = np.array(num_nonzero_row_arr)

            if config['use_group_lasso'] or config['use_wd']:
                # plot clustering results
                if config['use_wd'] and not (config['use_growl'] or config['use_group_lasso']):
                    pass
                else:
                    cluster_plot(num_cluster_arr, num_nonzero_row_arr, config)

                # get the group information
                _, _, group_info = display_similarity(sess, epoch+1, True, config)

                np.save(config['summary_dir']+'group_info.npy', np.array(group_info))
            # save the final trained model
            saver.save(sess, config['summary_dir']+'train_model/train_model_final.ckpt')

            #######################################################################
            # Retraining
            #######################################################################
            print('Now start retraining!')

            best_accuracy = 0
            # reset the momentum optimizer
            momentum_initializer = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
            sess.run(momentum_initializer)

            if config['retraining']:
                for epoch in range(config['num_epochs_retrain']):
                    for i in range(batch_per_epoch):
                        inputs, labels = mnist_data.train.next_batch(config['batch_size'])
                        # Run the optimizer, record the parameters if global_step % freq == 0
                        if i % config['tensorboard_freq'] == 0:

                            summary_extend = tf.Summary()

                            _, summary, retrain_loss = sess.run([optimizer_retrain, summary_op, loss],
                                feed_dict={X_image_flat: inputs,
                                Y_true: labels,
                                phase: True})

                            lr_rt = sess.run(lr_retrain)

                            #Add compression ratio and testing accuracy to tensorboard
                            summary_extend.ParseFromString(summary)
                            summary_extend.value.add(tag='retrain_testing_accuracy', simple_value=test_accur)
                            summary_extend.value.add(tag='retrain_compression_ratio', simple_value=compression_ratio)

                            step = sess.run(global_step_retrain)
                            writer.add_summary(summary_extend, step)

                        else:
                            _, retrain_loss = sess.run([optimizer_retrain, loss],
                                feed_dict={X_image_flat: inputs,
                                Y_true: labels,
                                phase: True})

                     # update the mask every epoch
                    if (config['use_growl'] | config['use_group_lasso']) & (epoch % config['mask_update_freq'] == 0):

                        # enforce the paramter sharing and keep pruning the parameters
                        # TODO: set a variable group and only update the group
                        compression_ratio, zero_layers = update_mask(sess, epoch, lr, mask_threshold, True, config, group_info, get_nonzero_idx_flag)

                    get_nonzero_idx_flag = 0

                    # testing
                    test_accur= sess.run(pred_accur,
                        feed_dict={X_image_flat: mnist_data.test.images,
                        Y_true: mnist_data.test.labels,
                        phase: False})

                    if test_accur > best_accuracy:
                        best_accuracy = test_accur

                    print('Epoch:{}, lr_rt={}, retrain_loss={:.4f}, accuracy={:.4f}'.format(epoch+1, lr_rt, retrain_loss, test_accur))

                    # store the testing accuracy and compression ratio
                    test_accur_arr.append(test_accur)
                    compression_ratio_arr.append(compression_ratio)

                    if ((epoch+1) % config['display_similarity'] == 0):

                        # storing the model
                        summary_dir = config['summary_dir'] + 'retrain_model/'
                        if not os.path.exists(summary_dir):
                            os.makedirs(summary_dir)
                        saver.save(sess, summary_dir, global_step=step)
                        display_similarity(sess, config['num_epochs']+epoch+1, False, config)

            if config['use_growl']:
                # plot the testing accuracy and the compression ratio
                accuracy_compression_plot(np.arange(config['num_epochs']+config['num_epochs_retrain']), test_accur_arr, compression_ratio_arr, config)

            # save results
            np.savetxt(config['plot_dir'] + 'accuracy.csv', test_accur_arr)
            np.savetxt(config['plot_dir'] + 'compression.csv', compression_ratio_arr)
            np.savetxt(config['plot_dir'] + 'best_accuracy.csv', list([best_accuracy]))
            if config['retraining'] == False:
                np.savetxt(config['plot_dir'] + 'best_accuracy.csv', [np.array(test_accur_arr).max()])
            saver.save(sess, config['summary_dir']+'retrain_model/retrain_model_final.ckpt')

    return test_accur, compression_ratio


if __name__ == '__main__':

    fc_layers = 3

    with open('config.yaml', 'r') as f:

        config = yaml.load(f)

        if not os.path.exists(config['plot_dir']):
            os.makedirs(config['plot_dir'])

        # record the training settings
        with open(config['plot_dir'] + 'config.yaml', 'w+') as record_config:
                    yaml.dump(config, record_config)

        train_mnist(config)
