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

from utils_plot import imagesc
import matplotlib
import matplotlib.pyplot as plt
from mnist import train_mnist


if __name__ == '__main__':

	layers = 5

	with open('config_search.yaml', 'r+') as f_search:
		config_search = yaml.load(f_search)

	lambda1_arr = np.linspace(1e-2, 1e0, num=4)
	lambda2_arr = np.linspace(1e-3, 2e-2, num=4)

	# run over different learning rate
	for lr_idx, lr_val in enumerate(config_search['learning_rate_vec']):
		config_search['learning_rate'] = lr_val

		# iterates through all combinations
		test_accur_arr = []
		compression_ratio_arr = []
		ca_metric_arr = []
		for lambda1_idx, lambda1 in enumerate(lambda1_arr):
			for lambda2_idx, lambda2 in enumerate(lambda2_arr):

				# update the regularization parameters
				reg_params = [[lambda1, lambda2]] * layers
				if config_search['use_growl']:
					config_search['growl_params'] = reg_params

				print("reg_params:{0}".format(reg_params))

				# create the directories for recording results and summaries
				subscript = str(lr_idx) + '-' + str(lambda1_idx) + '-' + str(lambda2_idx)
				config_search['summary_dir'] = config_search['summaries_dir'] + 'summary_' + subscript + '/'
				config_search['plot_dir'] = config_search['plots_dir'] + 'plot_' + subscript + '/'

				if not os.path.exists(config_search['plot_dir']):
					os.makedirs(config_search['plot_dir'])

				# record config info
				with open(config_search['plot_dir'] + 'config.yaml', 'w+') as f:
					yaml.dump(config_search, f)
					np.savetxt(config_search['plot_dir'] + 'reg_params.txt', reg_params)

				# run training and store the test accuracy for each run
				try:
					test_accur, compression_ratio = train_mnist(config_search)
					test_accur_arr.append(test_accur)
					compression_ratio_arr.append(compression_ratio)
				except KeyboardInterrupt:
					break


		plt.close()
		# x-axis is lambda1, y-axis is lambda2
		test_accur_map = np.reshape(test_accur_arr, (np.size(lambda1_arr), np.size(lambda2_arr))).T

		# grid
		x, y = np.meshgrid(lambda1_arr, lambda2_arr)

		# plot the grid search results for accuracy
		figName = config_search['plots_dir'] + 'accur_grid_search_{}.png'.format(lr_idx)
		imagesc(x, y, test_accur_map, 'lambda1', 'lambda2', 'Accuracy v.s. Reg Params, lr:{:.5f}'.format(lr_val),
		 False, figName)
		np.savetxt(config_search['plots_dir'] + 'accuracies.csv')

		# plot the grid search results for compression ratio
		figName = config_search['plots_dir'] + 'compression_grid_search_{}.png'.format(lr_idx)
		compression_ratio_map = np.reshape(compression_ratio_arr, (np.size(lambda1_arr), np.size(lambda2_arr))).T
		imagesc(x, y, compression_ratio_map, 'lambda1', 'lambda2', 'Compression ratio v.s. Reg Params, lr:{:.5f}'.format(lr_val),
		 False, figName)
		np.savetxt(config_search['plots_dir'] + 'compression_ratio.csv')

		# plot the ca metric
		figName = config_search['plots_dir'] + 'ca_search_{}.png'.format(lr_idx)
		ca_map = compression_ratio_map * test_accur_map
		imagesc(x, y, ca_map, 'lambda1', 'lambda2', 'Compression ratio v.s. Reg Params, lr:{:.5f}'.format(lr_val),
		 False, figName)
