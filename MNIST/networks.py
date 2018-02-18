import tensorflow as tf

from utils_nn import fully_connected_layer, relu_layer, convolutional_layer
from utils_nn import batch_normalization_layer, max_pool_2x2
import numpy as np

def LeNet_cnn(x_input, phase, gpu, config):

	# convolutional layer 1
	X_image = tf.reshape(x_input, [-1, 28, 28, 1])
	conv1 = convolutional_layer(X_image, [5,5,1,64], 'conv1', gpu, config)
	# max pooling layer 1
	max_pool1 = max_pool_2x2(conv1, 'max_pool1')
	# relu layer 1
	relu1 = relu_layer(max_pool1, 'relu1')
	# bn layer 1
	conv1_bn = batch_normalization_layer(relu1, axis=-1, phase=phase, name='bn1')

	# convolutional layer 2
	conv2 = convolutional_layer(conv1_bn, [5,5,64,32], 'conv2', gpu, config)
	# max pooling layer 2
	max_pool2 = max_pool_2x2(conv2, 'max_pool2')
	# relu layer 2
	relu2 = relu_layer(max_pool2, 'relu2')
	# bn layer 2
	conv2_bn = batch_normalization_layer(relu2, axis=-1, phase=phase, name='bn2')

	relu2_flat = tf.reshape(conv2_bn, [-1, np.prod(relu2.get_shape().as_list()[1:])])

	# fully connected layer 1
	# fc1 = fully_connected_layer(X_image_flat, 32, 'fc1', gpu, config)
	fc1 = fully_connected_layer(relu2_flat, 128, 'fc1', gpu, config)

	# relu layer 3
	relu3 = relu_layer(fc1, 'relu3')

	# batch normalization layer 3
	fc1_bn = batch_normalization_layer(relu3, axis=-1, phase=phase, name='bn3')

	# fully connected layer 2
	fc2 = fully_connected_layer(fc1_bn, 64, 'fc2', gpu, config)

	# relu layer 4
	relu4 = relu_layer(fc2, 'relu4')

	# batch normalization layer 2
	fc2_bn = batch_normalization_layer(relu4, axis=-1, phase=phase, name='bn4')

	# logits layer
	logits = fully_connected_layer(fc2_bn, 10, 'logits', gpu, config)


	return logits



def LeNet_300_100(x_input, phase, gpu, config):

	# fully connected layer 1
	fc1 = fully_connected_layer(x_input, 300, 'fc1', gpu, config)

	# relu layer 1
	relu1 = relu_layer(fc1, 'relu1')

	# batch normalization layer 1
	fc1_bn = batch_normalization_layer(relu1, axis=-1, phase=phase, name='bn1')

	# fully connected layer 2
	fc2 = fully_connected_layer(fc1_bn, 100, 'fc2', gpu, config)

	# relu layer 2
	relu2 = relu_layer(fc2, 'relu2')

	# batch normalization layer 1
	fc2_bn = batch_normalization_layer(relu2, axis=-1, phase=phase, name='bn2')

	# logits layer
	logits = fully_connected_layer(fc2_bn, 10, 'logits', gpu, config)


	return logits


def LeNet_single(x_input, phase, gpu, config):

	# fully connected layer 1
	fc1 = fully_connected_layer(x_input, 300, 'fc1', gpu, config)

	# relu layer 1
	relu1 = relu_layer(fc1, 'relu1')

	# batch normalization layer 1
	fc1_bn = batch_normalization_layer(relu1, axis=-1, phase=phase, name='bn1')

	# logits layer
	logits = fully_connected_layer(fc1_bn, 10, 'logits', gpu, config)


	return logits













