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
from utils_GrOWL import reg_params_init, apply_reg_prox, update_mask, measure_compression, adjust_learning_rate, preprocess_hparam, set_param_share
from utils_retrain import display_similarity, apply_param_share
from vgg_model import vgg16

BATCH_PER_EPOCH = int(FLAGS.num_training_examples_per_epoch / FLAGS.batch_size)
TRAIN_STEPS = BATCH_PER_EPOCH * FLAGS.num_training_epochs
RETRAIN_STEPS = BATCH_PER_EPOCH * FLAGS.num_retraining_epochs
EVAL_BATCHES = int(10000/100)
REG_APPLY_FREQ = FLAGS.reg_apply_epoch_freq * BATCH_PER_EPOCH


def train(hps, res_dict):

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            train_images, train_labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, 128, 'train')
            test_images, test_labels = cifar_input.build_input(FLAGS.dataset, FLAGS.eval_data_path, 100, 'eval')

        lr = tf.placeholder(tf.float32)
        phase = tf.placeholder(tf.bool) # true for training
        x_input = tf.cond(phase, lambda:train_images, lambda:test_images)
        Y_true = tf.cond(phase, lambda:train_labels, lambda:test_labels)

        loss, precision = vgg16(x_input, Y_true, phase, True)
        tf.summary.scalar('Precision', precision)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            grads_and_vars = opt.compute_gradients(loss)
            optimizer = opt.apply_gradients(grads_and_vars, global_step=global_step)

        # Add histograms for gradients
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Summary op
        summary_op = tf.summary.merge_all()

        # Initializer for the variables
        init = tf.global_variables_initializer()

        # Saver op
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(FLAGS.res_dir, graph=tf.get_default_graph())

        train_learning_rate = FLAGS.training_lr_init
        TEST_ACCURACY=0.1
        compression_ratio = 1.0
        zero_layers=0
        with tf.Session() as sess:

            if FLAGS.resume:
                # restore the model
                try:
                    ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_dir)
                except tf.errors.OutOfRangeError as e:
                    tf.logging.error('Cannot restore checkpoint: %s', e)

                if not (ckpt_state and ckpt_state.model_checkpoint_path):
                    tf.logging.info('No model to load yet at %s', FLAGS.train_dir)

                tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
                saver.restore(sess, ckpt_state.model_checkpoint_path)
                model_step = ckpt_state.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print("\n Resume model was saved at step {}".format(model_step))
                print('train steps:{}, model step:{}'.format(TRAIN_STEPS, model_step))
                train_learning_rate = FLAGS.resume_lr_init
            else:
                sess.run(init)

            # initialize regularization parameter
            if FLAGS.use_growl | FLAGS.use_group_lasso:
                layer_reg_params, hps = reg_params_init(sess, hps)

            # Start input enqueue threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                prev_time = time.clock()
                while not coord.should_stop():
                    step = sess.run(global_step)

                    # check whether to decay learning rate
                    if (step+1) % FLAGS.lr_decay_step == 0:
                        train_learning_rate = train_learning_rate * FLAGS.lr_decay_rate

                    if step % BATCH_PER_EPOCH == 0:
                        summary_extend = tf.Summary()
                        _, summary, train_loss, train_accuracy = sess.run([optimizer, summary_op, loss, precision],
                                                                           feed_dict={lr:train_learning_rate, phase:True})
                        summary_extend.ParseFromString(summary)
                        summary_extend.value.add(tag='testing accuracy', simple_value=TEST_ACCURACY)
                        summary_extend.value.add(tag='compression', simple_value=compression_ratio)
                        writer.add_summary(summary_extend, step)
                        print('step: {}, lr_rate: {}, train_loss: {:.4f}, train_accuracy: {:.4f}'.format(step, train_learning_rate, train_loss, train_accuracy))
                    else:
                        sess.run([optimizer], feed_dict={lr:train_learning_rate,phase:True})

                    # save the model and evaluate
                    if (step % FLAGS.checkpoint_freq == 0 and step != 0) or (step == TRAIN_STEPS):
                        print('Checkpoint! Now saving model...')
                        saver.save(sess, FLAGS.train_dir, global_step=step)

                    if step % FLAGS.eval_freq == 0:
                        current_time = time.clock()
                        test_loss = 0
                        test_accuracy = 0
                        for i in range(EVAL_BATCHES):
                            test_img_vals = sess.run(test_images)
                            test_label_vals = sess.run(test_labels)
                            test_loss_i, test_accur_i = sess.run([loss, precision], feed_dict={phase:False})
                            test_loss += test_loss_i
                            test_accuracy += test_accur_i

                        test_loss = test_loss / EVAL_BATCHES
                        TEST_ACCURACY = test_accuracy = test_accuracy / EVAL_BATCHES
                        res_dict['test_accur_arr'].append(test_accuracy)
                        res_dict['training_accur_arr'].append(train_accuracy)
                        res_dict['steps'].append(step)
                        batch_time = (current_time-prev_time)/FLAGS.checkpoint_freq
                        print('    TEST_ACCURACY: {:.4f}, 1 batch takes: {:.4f}'.format(test_accuracy, batch_time))
                        prev_time = current_time

                    # apply proximal gradient update, and update the mask
                    if (step % REG_APPLY_FREQ==0) and (step>0) and FLAGS.use_sparse_reg:
                        apply_reg_prox(sess, train_learning_rate, layer_reg_params, hps)

                        # update mask
                        zero_layers, layer_ID = update_mask(sess, FLAGS.mask_threshold, hps, res_dict, step)
                        compression_ratio = measure_compression(sess, res_dict, step, True, hps)

                        if zero_layers >= 1:
                            print("There exists zero value layers at step:{0}, layers IDs:{1}".format(step, layer_ID))
                            coord.request_stop()

                    if ((step >= TRAIN_STEPS) and FLAGS.retrain_on) or ((step % FLAGS.display_similarity_freq==0) and step>1):
                        print("Get the group information! \n")
                        group_info, num_clusters_arr = display_similarity(sess, FLAGS.num_training_epochs, hps, res_dict)
                        np.save(FLAGS.train_dir + 'group_info.npy', group_info)
                        np.save(FLAGS.train_dir + 'num_clusters_arr.npy', num_clusters_arr)

                    if step >= TRAIN_STEPS:
                        coord.request_stop()

            except tf.errors.OutOfRangeError:
                np.save(FLAGS.res_dir + 'res_dict.npy', res_dict)
                print('Done training')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

            return zero_layers, step


def retrain(hps, res_dict):

    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            train_images, train_labels = cifar_input.build_input(FLAGS.dataset, FLAGS.train_data_path, 128, 'train')
            test_images, test_labels = cifar_input.build_input(FLAGS.dataset, FLAGS.eval_data_path, 100, 'eval')

        lr = tf.placeholder(tf.float32)
        phase = tf.placeholder(tf.bool) # true for training
        x_input = tf.cond(phase, lambda:train_images, lambda:test_images)
        Y_true = tf.cond(phase, lambda:train_labels, lambda:test_labels)

        loss, precision = vgg16(x_input, Y_true, phase, True)
        tf.summary.scalar('Precision', precision)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            grads_and_vars = opt.compute_gradients(loss)
            optimizer = opt.apply_gradients(grads_and_vars, global_step=global_step)

        # Add histograms for gradients
        for grad, var in grads_and_vars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # Summary op
        summary_op = tf.summary.merge_all()

        # Initializer for the variables
        init = tf.global_variables_initializer()

        # Saver op
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(FLAGS.res_dir, graph=tf.get_default_graph())

        train_learning_rate = FLAGS.retraining_lr_init

        TEST_ACCURACY=0.1
        compression_ratio=1
        with tf.Session() as sess:

            # restore the model
            try:
                ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_dir)
            except tf.errors.OutOfRangeError as e:
                tf.logging.error('Cannot restore checkpoint: %s', e)

            if not (ckpt_state and ckpt_state.model_checkpoint_path):
                tf.logging.info('No model to load yet at %s', FLAGS.train_dir)

            tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            model_step = ckpt_state.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print("Now start retraining process, resotred model was saved at step {}".format(model_step))
            print('train steps:{}, model step:{}'.format(TRAIN_STEPS, model_step))
            # assert model_step == TRAIN_STEPS
            momentum_initializer = [var.initializer for var in tf.global_variables() if 'Momentum' in var.name]
            sess.run(momentum_initializer)
            
            #load cluster info
            group_info = np.load(FLAGS.train_dir + 'group_info.npy')
            num_cluster_arr = np.load(FLAGS.train_dir + 'num_clusters_arr.npy')
            
            # Start input enqueue threads
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                prev_time = time.clock()
                while not coord.should_stop():

                    step = sess.run(global_step)

                    # check whether to decay learning rate
                    if (step+1 - int(model_step)) % FLAGS.retrain_lr_decay_step == 0:
                        train_learning_rate = train_learning_rate * FLAGS.lr_decay_rate

                    if step % BATCH_PER_EPOCH == 0:
                        summary_extend = tf.Summary()
                        _, summary, train_loss, train_accuracy = sess.run([optimizer, summary_op, loss, precision],
                                                                           feed_dict={lr:train_learning_rate, phase:True})
                        summary_extend.ParseFromString(summary)
                        summary_extend.value.add(tag='testing accuracy', simple_value=TEST_ACCURACY)
                        summary_extend.value.add(tag='compression', simple_value=compression_ratio)
                        writer.add_summary(summary_extend, step)
                        print('step: {}, lr_rate: {}, train_loss: {:.4f}, train_accuracy: {:.4f}'.format(step, train_learning_rate, train_loss, train_accuracy))
                    else:
                        sess.run([optimizer], feed_dict={lr:train_learning_rate,phase:True})

                    # save the model and evaluate
                    if step % FLAGS.checkpoint_freq_retrain == 0 and step != 0:
                        print('Checkpoint! Now saving model...')
                        saver.save(sess, FLAGS.retrain_dir, global_step=step)

                    if step % FLAGS.eval_freq == 0:
                        current_time = time.clock()
                        test_loss = 0
                        test_accuracy = 0
                        for i in range(EVAL_BATCHES):
                            test_img_vals = sess.run(test_images)
                            test_label_vals = sess.run(test_labels)
                            test_loss_i, test_accur_i = sess.run([loss, precision], feed_dict={phase:False})
                            test_loss += test_loss_i
                            test_accuracy += test_accur_i

                        test_loss = test_loss / EVAL_BATCHES
                        TEST_ACCURACY = test_accuracy = test_accuracy / EVAL_BATCHES
                        res_dict['test_accur_arr'].append(test_accuracy)
                        res_dict['training_accur_arr'].append(train_accuracy)
                        res_dict['steps'].append(step)
                        batch_time = (current_time-prev_time)/FLAGS.checkpoint_freq
                        print('    TEST_ACCURACY: {:.4f}, 1 batch takes: {:.4f}'.format(test_accuracy, batch_time))
                        prev_time = current_time

                    #Retraining phase: parameter sharing
                    if FLAGS.param_share:
                        apply_param_share(sess, group_info, hps)

                    # measure compression ratio
                    if (step % REG_APPLY_FREQ==0) and (step>0) and FLAGS.use_sparse_reg:
                        compression_ratio=measure_compression(sess, res_dict, step, False, hps, num_cluster_arr=num_cluster_arr)

                    if step >= TRAIN_STEPS + RETRAIN_STEPS:
                        print("\n Ending the retraining phase!!")
                        coord.request_stop()

            except tf.errors.OutOfRangeError:
                np.save(FLAGS.res_dir + 'res_dict.npy', res_dict)
                print('Done retraining')
            finally:
                coord.request_stop()

            coord.join(threads)
            sess.close()

def main(argv=None):

    batch_size = 128
    num_classes = 10

    #Create heyperparameters and results dictionary
    reg_params=np.array([[100, 100],[4.5, 1e-1],[11, 1e-1],[11, 1e-1],[11, 1e-1],
                        [10, 1e-1],[9, 1e-1],[7.5, 1e-1],[7.0, 1e-1],[6.5, 7e-2],
                        [5.5, 5e-2],[5.5, 1e-1],[3.5, 1e-1],[3.5, 1e-1],[100,100]])

    # 1 and 15 no reg, 2-14 reg
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

    zero_layers, step = train(hps, res_dict)
    if zero_layers and step <= 390*60:
        print("there existed zero valued layers at step:{}".format(step))
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

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
