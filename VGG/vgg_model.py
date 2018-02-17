from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from flags import FLAGS
from utils_nn import ConvBNReLU, fully_connected_layer, max_pool_2x2, batch_normalization_layer

def vgg16(x_input, Y_true, phase, gpu):

    h = ConvBNReLU(x=x_input, filter_spec=[3,3,3,64], axis=-1, phase=phase, name='c1', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,64,64], axis=-1, phase=phase, name='c2', gpu=True)
    h = max_pool_2x2(h, 'mp1')

    h = ConvBNReLU(x=h, filter_spec=[3,3,64,128], axis=-1, phase=phase, name='c3', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,128,128], axis=-1, phase=phase, name='c4', gpu=True)
    h = max_pool_2x2(h, 'mp2')

    h = ConvBNReLU(x=h, filter_spec=[3,3,128,256], axis=-1, phase=phase, name='c5', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,256,256], axis=-1, phase=phase, name='c6', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,256,256], axis=-1, phase=phase, name='c7', gpu=True)
    h = max_pool_2x2(h, 'mp3')

    h = ConvBNReLU(x=h, filter_spec=[3,3,256,512], axis=-1, phase=phase, name='c8', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,512,512], axis=-1, phase=phase, name='c9', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,512,512], axis=-1, phase=phase, name='c10', gpu=True)
    h = max_pool_2x2(h, 'mp4')

    h = ConvBNReLU(x=h, filter_spec=[3,3,512,512], axis=-1, phase=phase, name='c11', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,512,512], axis=-1, phase=phase, name='c12', gpu=True)
    h = ConvBNReLU(x=h, filter_spec=[3,3,512,512], axis=-1, phase=phase, name='c13', gpu=True)
    h = max_pool_2x2(h, 'mp5')

    h = tf.reshape(h, [-1, np.prod(h.get_shape().as_list()[1:])])
    h = fully_connected_layer(h, 512, 'fc1', gpu=True)
    h = batch_normalization_layer(h, axis=-1, phase=phase, name='fc1/bn1')

    # calculate loss
    logits = fully_connected_layer(h, 10, 'logits', gpu=True)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_true, logits=logits, name='cross_entropy'),name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy)
    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y_true,1))
    precision = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('training_accuracy', precision)

    return loss, precision
