from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
from flags import FLAGS

def _weights_initializer(name, shape, stddev, gpu):

	if gpu:
		with tf.device('/gpu:0'):
			if len(np.shape(shape)) == 2:
				weights = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=stddev), dtype=tf.float32)
			else:
				weights = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=stddev), dtype=tf.float32)

			if FLAGS.use_weight_decay:
				weight_decay = tf.multiply(tf.nn.l2_loss(weights), FLAGS.weight_decay_rate, name='weight_loss')
				tf.add_to_collection('losses', weight_decay)
	else:
		with tf.device('/cpu:0'):
			weights = tf.get_variable(name=name,
					shape=shape,
					initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
					dtype=tf.float32)
			print('on cpu')

	return weights

def _biases_initializer(name, shape, val, gpu):
	if gpu:
		with tf.device('/gpu:0'):
			biases = tf.get_variable(name=name,
				shape=shape,
				initializer=tf.constant_initializer(val),
				dtype=tf.float32)

			if FLAGS.use_weight_decay:
				weight_decay = tf.multiply(tf.nn.l2_loss(biases), FLAGS.weight_decay_rate, name='weight_loss')
				tf.add_to_collection('losses', weight_decay)
	else:
		with tf.device('/cpu:0'):
			biases = tf.get_variable(name=name,
				shape=shape,
				initializer=tf.constant_initializer(val),
				dtype=tf.float32)

	return biases

def _mask_initializer(name, shape, gpu):
	'''
	Mask is not going to be trained, we need to set trainable to False
	'''
	if gpu:
		with tf.device('/gpu:0'):
			mask = tf.get_variable(name=name,
				shape=shape,
				initializer=tf.ones_initializer(),
				trainable=False)
	else:
		with tf.device('/cpu:0'):
			mask = tf.get_variable(name=name,
				shape=shape,
				initializer=tf.ones_initializer(),
				trainable=False)
	return mask

def _parameter_summary(params):
	tf.summary.histogram(params.op.name, params)
	tf.summary.histogram(params.op.name + '/row_norm', tf.reduce_sum(tf.pow(tf.norm(params, axis=(0,1)), 2), axis=1))
	tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

def _parameter_summary_fc(params):
	tf.summary.histogram(params.op.name, params)
	tf.summary.histogram(params.op.name + '/row_norm', tf.pow(tf.norm(params, axis=1), 2))
	tf.summary.scalar(params.op.name + '/spartisty', tf.nn.zero_fraction(params))


def _image_summary(params):
	tf.summary.image(params.op.name, params, max_outputs=1)

def _output_summary(outputs):
	tf.summary.histogram(outputs.op.name + '/outputs', outputs)
	tf.summary.scalar(outputs.op.name + '/outputs_sparsity',
		tf.nn.zero_fraction(outputs))

def get_weight_placeholders():

    return tf.get_collection('weight_placeholder')

def get_mask_placeholders():

    return tf.get_collection('mask_placeholder')



#################################################################################
#LAYERS
#################################################################################
def fully_connected_layer(x, units, name, gpu):
	'''
	Args:
		x: inputs from the previous layer, shape = [None, x.shape[0]].
		units: # of nodes of this layer, scalar.
		name: the name of this layer in graph.
		config: yaml file for the configuration.

	Returns:
		h: outputs, 1D tensor, shape = [None, units]
	'''
	with tf.variable_scope(name) as scope:

		# initialize layer parameters
		weights = _weights_initializer(name='weights',
			shape=[x.get_shape().as_list()[1], units],
			stddev=np.sqrt(2/units),
			gpu=gpu)
		biases = _biases_initializer(name='biases',
			shape=[units],
			val=0.1,
			gpu=gpu)

		# initilize and apply mask if specified to
		if FLAGS.use_mask:
			mask = _mask_initializer(name='mask',
				shape=weights.get_shape().as_list(),
				gpu=gpu)
			# Apply mask to the weights
			masked_weights = tf.multiply(mask, weights, name='masked_weights')

			_parameter_summary_fc(masked_weights)
			new_shape = masked_weights.get_shape().as_list()
			new_shape.insert(0,1)
			new_shape.append(1)
			_image_summary(tf.reshape(masked_weights, new_shape))

			# Calculate the output of the fully connected layer
			h = tf.add(tf.matmul(x, masked_weights), biases, name=name)
			_parameter_summary_fc(mask)
		else:
			h = tf.add(tf.matmul(x, weights), biases, name=name)

		# Adds summary to the weights and outputs
		_parameter_summary_fc(weights)
		_output_summary(h)

		if FLAGS.use_growl | FLAGS.use_group_lasso:
			# create interface for OWL
			w_placeholder = tf.placeholder(tf.float32, [x.get_shape().as_list()[1], units])
			assign_op_w = tf.assign(weights, w_placeholder, validate_shape=True)
			tf.add_to_collection('weight_placeholder', (weights, w_placeholder, assign_op_w))

			mask_placeholder = tf.placeholder(tf.float32, [x.get_shape().as_list()[1], units])
			assign_op_m = tf.assign(mask, mask_placeholder, validate_shape=True)
			tf.add_to_collection('mask_placeholder', (mask, mask_placeholder, assign_op_m))

	return h

def convolutional_layer(x, filter_spec, name, gpu):
	'''
	Args:
		x: inputs, a 4d tensor.
		filter: 4-tuple, [filter_height, filter_width, in_channels, out_channels].
		name: name for this layer.

	Returns:
		h: convoluted results.
	'''
	filter_height = filter_spec[0]
	filter_width = filter_spec[1]
	in_channels = filter_spec[2]
	out_channels = filter_spec[3]
	n = filter_height * filter_width * out_channels

	with tf.variable_scope(name):
		weights = _weights_initializer(name='weights',
			shape=filter_spec,
			stddev=np.sqrt(2/n),
			gpu=gpu)
		biases = _biases_initializer(name='biases',
			shape=filter_spec[3],
			val=0.1,
			gpu=gpu)

		# initilize and apply mask if specified to
		if FLAGS.use_mask:
			mask = _mask_initializer(name='mask',
				shape=filter_spec,
				gpu=gpu)
			# Apply mask to the weights
			masked_weights = tf.multiply(mask, weights, name='masked_weights')
			_parameter_summary(masked_weights)
			
			h = tf.add(tf.nn.conv2d(x, masked_weights, strides=[1,1,1,1], padding='SAME'), biases, name=name)
		else:
			h = tf.add(tf.nn.conv2d(x, weights, strides=[1,1,1,1], padding='SAME'), biases, name=name)

		# Add summary for the weights and outputs
		_parameter_summary(weights)
		_output_summary(h)

		# Add interface for OWL/grOWL
		if FLAGS.use_growl | FLAGS.use_group_lasso:
			w_placeholder = tf.placeholder(tf.float32, filter_spec)
			assign_op_w = tf.assign(weights, w_placeholder, validate_shape=True)
			tf.add_to_collection('weight_placeholder', (weights, w_placeholder, assign_op_w))

			mask_placeholder = tf.placeholder(tf.float32, filter_spec)
			assign_op_m = tf.assign(mask, mask_placeholder, validate_shape=True)
			tf.add_to_collection('mask_placeholder', (mask, mask_placeholder, assign_op_m))


	return h

def relu_layer(x, name):
	'''
	Args:
		x: inputs from the previous layer.

	Returns:
		h: activation.
	'''
	with tf.variable_scope(name) as scope:
		h = tf.nn.relu(x, name=name)
		_output_summary(h)

	return h

def max_pool_2x2(x, name):
	'''
	Args:
		x:inputs.

	Returns:
		h: results.
	'''
	h = tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
	return h

def batch_normalization_layer(x, axis, phase, name):
	'''
	Args:
		x: inputs.
		phase: boolean, true for training, false for testing.

	Returns:
		h: results.
	'''

	h = tf.layers.batch_normalization(x, axis=axis, training=phase, name=name)

	return h

def ConvBNReLU(x, filter_spec, axis, phase, name, gpu):
	'''
	Composite layer for convolution, batch_normalization, and ReLU
	Args:
	    x: inputs
	    filter_spec: filters specs for convolutional filter
	    axis: the axis for applying normalization
	    phase: true for training, false for testing
	    name: name for the layer
	    gpu:

	Returns:
	    h: results
	'''
	with tf.variable_scope(name+'conv'):
		h = convolutional_layer(x, filter_spec, name+'/conv', gpu)

	with tf.variable_scope(name+'BN'):
		h = batch_normalization_layer(h, axis, phase, name+'/BN')

	with tf.variable_scope(name+"ReLE"):
		h = relu_layer(h, name+'/ReLU')

	return h
