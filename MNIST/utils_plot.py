'''
This module contains functions useful for plotting resutls
1. imagesc
2.
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


def imagesc(x, y, z, xlabel, ylabel, title, log, name):
	'''
	Plot a 2d color map image
	'''
	if np.size(z) == 0:
		print('z is empty, no figure is printed')
		return

	fig = plt.pcolormesh(x, y, z, cmap='jet', vmax=1, vmin=-1)
	# fig = plt.pcolormesh(x, y, z, cmap='jet')
	plt.colorbar()
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	if log:
		plt.xscale('log')
		plt.yscale('log')

	plt.savefig(name)
	plt.close()

def cluster_plot(num_cluster_arr_all, num_nonzero_row_arr_all, config):
	'''
	Plot the clustering resutls

	Input:
		num_cluster_arr_all: number of cluster array for all layers, each element is an array which
		contains the numbers of clusters for all the regularized layers

	'''

	# get the number of layers we applied grOWL on
	# num_layer = np.sum(config['owl_applied_layers'] == True)

	layer_switches = config['owl_applied_layers']

	idx = 0

	for idx_layer, switch in enumerate(layer_switches):

		if switch == False:
			continue

		# read the results for the corresponding layer
		num_cluster_arr = num_cluster_arr_all[:, idx]
		num_nonzero_row_arr = num_nonzero_row_arr_all[:, idx]

		# plot the nonzero rows and number of clusters
		fig, ax1 = plt.subplots()
		x_ticks = np.arange(1, config['num_epochs']+config['display_similarity']+1, config['display_similarity'])
		if config['display_similarity'] == 1:
			x_ticks = np.arange(1, config['num_epochs']+config['display_similarity'], config['display_similarity'])
		nonzero_plot, = ax1.plot(x_ticks, num_nonzero_row_arr, 'bo-', label='nonzero')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Number of rows',color='b')
		ax1.tick_params('y', colors='b')


		ax2 = ax1.twinx()
		cluster_plot, = ax2.plot(x_ticks, num_cluster_arr, 'r*-', label='cluster')
		ax2.set_ylabel('Number of clusters', color='r')
		ax2.tick_params('y', colors='r')

		plt.legend(handles=[nonzero_plot, cluster_plot])

		fig.savefig(config['plot_dir'] + 'correlation_layer{}.png'.format(idx+1))
		plt.close()

		# plot the nonzero/cluster raito, or the average cluster size
		fig, ax1 = plt.subplots()
		avgsize_plot, = ax1.plot(x_ticks, np.divide(num_nonzero_row_arr, num_cluster_arr), 'g^-', label='avgsize')
		ax1.set_xlabel('Epochs')
		ax1.set_ylabel('Average cluster size')
		fig.savefig(config['plot_dir'] + 'avgsize_layer{}.png'.format(idx+1))
		plt.close()

		idx = idx + 1

def accuracy_compression_plot(x_ticks, test_accur_arr, compression_ratio_arr, config):

	# if config['retraining']:
	# 	x_ticks = np.arange(1, config['num_epochs']+config['num_epochs_retrain']+config['display_similarity']+1, config['display_similarity'])
	# else:
	# 	x_ticks = np.arange(1, config['num_epochs']+config['display_similarity']+1, config['display_similarity'])

	fig, ax1 = plt.subplots()
	testing_accur_plot, = plt.plot(x_ticks, test_accur_arr, 'r-', label='testing_accuracy')
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Testing accuracies', color='r')
	ax1.tick_params('y', colors='r')

	ax2 = ax1.twinx()
	compression_ratio_plot, = ax2.plot(x_ticks, np.multiply(compression_ratio_arr, 100), 'g-', label='compresion_ratio')
	ax2.set_ylabel('Compression Ratio %', color='g')
	ax2.tick_params('y', colors='g')

	plt.legend(loc = 'lower left', handles=[testing_accur_plot, compression_ratio_plot])

	plt.savefig(config['plot_dir'] + 'accuracy_compression.png')
	plt.close()


def hist_plot(idx, epoch, phase, row_norm, config):
    if not phase:
        np.savetxt(config['plot_dir'] + 'layer_{}_train_{}.csv'.format(idx, epoch+1), row_norm)
        plt.hist(row_norm)
        plt.title('Histogram of the row norms of layer {} at train {} epochs'.format(idx, epoch+1))
        plt.savefig(config['plot_dir'] + 'layer{}_train_hist_{}.jpg'.format(idx, epoch+1))
        plt.close()
    else:
        np.savetxt(config['plot_dir'] + 'retrain_row_norm_{}.csv'.format(idx,epoch+1), row_norm)
        plt.hist(row_norm)
        plt.title('Histogram of the row norms of layer {} at retrain {} epochs'.format(idx,epoch+1))
        plt.savefig(config['plot_dir'] +  'layer{}_retrain_hist_{}.jpg'.format(idx,epoch+1))
        plt.close()
