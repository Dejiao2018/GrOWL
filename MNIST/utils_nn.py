'''
This module contains layers and useful funcitons for neural network
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
#################################################################################
'''
FUNCTIONS FOR NEURAL NETWORKS
	1. _weights_initial
	2. _biases_initial
	3. _mask_initializer
	4. _parameter_summary
	5. _output_summary
	6. get_weight_placeholders
	7. get_mask_placeholders

LAYERS FOR NEURAL NETWORKS
	1. batch_normalization_layer
	2. fully_connected_layer
	3. convolutional_layer
	4. relu_layer
	5. leaky relu layer
	6. max_pooling_layer
'''
#################################################################################


#################################################################################
#FUNCTIONS
#################################################################################
def _weights_initializer(name, shape, stddev, gpu, config):

	if gpu:
		with tf.device('/gpu:0'):
			if len(np.shape(shape)) == 2:
			# weights = tf.get_variable(name=name,
			# 		shape=shape,
			# 		initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
			# 		dtype=tf.float32)
				# weights = tf.get_variable(name=name, shape=shape, initializer=tf.uniform_unit_scaling_initializer(factor=1.0), dtype=tf.float32)
				weights = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=stddev), dtype=tf.float32)
			else:
				weights = tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(stddev=stddev), dtype=tf.float32)

			if config['use_wd']:
				weight_decay = tf.multiply(tf.nn.l2_loss(weights), config['wd'], name='weight_loss')
				tf.add_to_collection('losses', weight_decay)

	else:
		with tf.device('/cpu:0'):
			weights = tf.get_variable(name=name,
					shape=shape,
					initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
					dtype=tf.float32)
			print('on cpu')

	return weights

def _biases_initializer(name, shape, val, gpu, config):
	if gpu:
		with tf.device('/gpu:0'):
			biases = tf.get_variable(name=name,
				shape=shape,
				initializer=tf.constant_initializer(val),
				dtype=tf.float32)

			if config['use_wd']:
				weight_decay = tf.multiply(tf.nn.l2_loss(biases), config['wd'], name='weight_loss')
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
	tf.summary.histogram(params.op.name + '/spartisty', tf.nn.zero_fraction(params))

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
def fully_connected_layer(x, units, name, gpu, config):
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
			gpu=gpu,
			config=config)
		biases = _biases_initializer(name='biases',
			shape=[units],
			val=0.1,
			gpu=gpu,
			config=config)

		# initilize and apply mask if specified to
		if config['use_mask']:
			mask = _mask_initializer(name='mask',
				shape=weights.get_shape().as_list(),
				gpu=gpu)
			# Apply mask to the weights
			masked_weights = tf.multiply(mask, weights, name='masked_weights')

			_parameter_summary(masked_weights)
			new_shape = masked_weights.get_shape().as_list()
			new_shape.insert(0,1)
			new_shape.append(1)
			_image_summary(tf.reshape(masked_weights, new_shape))

			# Calculate the output of the fully connected layer
			h = tf.add(tf.matmul(x, masked_weights), biases, name=name)
		else:
			h = tf.add(tf.matmul(x, weights), biases, name=name)

		# Adds summary to the weights and outputs
		_parameter_summary(weights)

		_parameter_summary(mask)
		_output_summary(h)

		if config['use_owl'] | config['use_growl'] | config['use_group_lasso']:
			# create interface for OWL
			w_placeholder = tf.placeholder(tf.float32, [x.get_shape().as_list()[1], units])
			assign_op_w = tf.assign(weights, w_placeholder, validate_shape=True)
			tf.add_to_collection('weight_placeholder', (weights, w_placeholder, assign_op_w))

			mask_placeholder = tf.placeholder(tf.float32, [x.get_shape().as_list()[1], units])
			assign_op_m = tf.assign(mask, mask_placeholder, validate_shape=True)
			tf.add_to_collection('mask_placeholder', (mask, mask_placeholder, assign_op_m))

	return h

def convolutional_layer(x, filter_spec, name, gpu, config):
	'''
	Args:
		x: inputs, a 4d tensor.
		filter: 4-tuple, [filter_height, filter_width, in_channels, out_channels].
		name: name for this layer.

	Returns:
		h: convoluted results.
	'''
	n = filter_height * filter_width * in_channels
	with tf.variable_scope(name) as scope:
		weights = _weights_initializer(name='weights',
			shape=filter_spec,
			stddev=np.sqrt(2/n),
			gpu=gpu,
			config=config)
		biases = _biases_initializer(name='biases',
			shape=filter_spec[3],
			val=0.1,
			gpu=gpu,
			config=config)

		# initilize and apply mask if specified to
		if config['use_mask']:
			mask = _mask_initializer(name='mask',
				shape=filter_spec,
				gpu=gpu)
			# Apply mask to the weights
			masked_weights = tf.multiply(mask, weights, name='masked_weights')

			_parameter_summary(masked_weights)
			# new_shape = masked_weights.get_shape().as_list()
			# new_shape.insert(0,1)
			# new_shape.append(1)
			# _image_summary(tf.reshape(masked_weights, new_shape))

			# Calculate the output of the convolutional layer
			h = tf.add(tf.nn.conv2d(x, masked_weights, strides=[1,1,1,1], padding='SAME'), biases, name=name)
		else:
			h = tf.add(tf.nn.conv2d(x, weights, strides=[1,1,1,1], padding='SAME'), biases, name=name)

		# Add summary for the weights and outputs
		_parameter_summary(weights)
		_output_summary(h)

		# Add interface for OWL/grOWL
		if config['use_owl'] | config['use_growl'] | config['use_group_lasso'] | config['use_mask']:
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
