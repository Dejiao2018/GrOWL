import tensorflow as tf
from utils_nn import fully_connected_layer, relu_layer
from utils_nn import batch_normalization_layer
import numpy as np


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








