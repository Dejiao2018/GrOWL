import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
Code for generating fig 5

To run this code, please modify the log_root and folder_name accordingly
'''


# calculate the variance ratio
def var_ratio(sequences):
	mean_vector = np.mean(sequences, axis=0)
	diff_mat = sequences - mean_vector
	var_ratio = np.mean(norm(diff_mat, ord=1, axis=1))/np.count_nonzero(mean_vector)
	return var_ratio


log_root = '../paper_results'
titles = ['(a) Group Lasso (2)', '(b) GrOWL']
num_files = 5
sequences = []
fontsize=36
fontdict = {'fontsize':fontsize}

# Group Lasso
for i in range(0,num_files):
    folder_name = '/group_lasso_lr_0.001_l1_p_60_0.6_idx1_{}'.format(i)
    indices = np.load(log_root+folder_name+'/plot_resnonzero_row_idx.npy')
    sequence = np.zeros(784)
    sequence[indices] = 1
    sequences.append(sequence)

var_ratio0 = var_ratio(sequences)
print(var_ratio0)


sequences = np.array(sequences).T

fig, ax  = plt.subplots(4, 1,sharex=True, figsize=(40,16), dpi=100)
ax[0].spy(sequences.T, aspect=15, markersize=3, c='b')
x_axis0 = ax[0].get_xaxis()

x_axis0.set_ticks_position('bottom')

ax[0].set_xlabel('Row index', fontsize=fontsize)
ax[0].set_xticklabels(range(-100,800,100), fontdict=fontdict)
ax[0].set_yticklabels(range(6,0,-1), fontdict=fontdict)


# Grlasso+l2
sequences = []
num_files = 5
for i in range(0,num_files):
    folder_name = '/grEN_l1_80_l_wd_0.001_p_0.6_idx_{}'.format(i)
    indices = np.load(log_root+folder_name+'/plot_resnonzero_row_idx.npy')
    sequence = np.zeros(784)
    sequence[indices] = 1
    sequences.append(sequence)

var_ratio1 = var_ratio(sequences)
print(var_ratio1)

sequences = np.array(sequences).T


ax[2].spy(sequences.T, aspect=15, markersize=3, origin='lower', c='b')
x_axis1 = ax[2].get_xaxis()

x_axis1.set_ticks_position('bottom')

ax[2].set_xlabel('Row index', fontsize=fontsize)
ax[2].set_xticklabels(range(-100,800,100), fontdict=fontdict)
ax[2].set_yticklabels(range(6,0,-1), fontdict=fontdict)


# GrOWL
sequences = []
num_files = 5
for i in range(0,num_files):
    folder_name = '/PLDwd_lr_0.001_l1_12_l2_0.215_PLDt_0.5_p_0.6_wd_0_idx_{}'.format(i)
    indices = np.load(log_root+folder_name+'/plot_resnonzero_row_idx.npy')
    sequence = np.zeros(784)
    sequence[indices] = 1
    sequences.append(sequence)

var_ratio1 = var_ratio(sequences)
print(var_ratio1)

sequences = np.array(sequences).T


ax[1].spy(sequences.T, aspect=15, markersize=3, origin='lower', c='b')
x_axis1 = ax[1].get_xaxis()

x_axis1.set_ticks_position('bottom')

ax[1].set_xlabel('Row index', fontsize=fontsize)
ax[1].set_xticklabels(range(-100,800,100), fontdict=fontdict)
ax[1].set_yticklabels(range(6,0,-1), fontdict=fontdict)


# GrOWL +l2
sequences = []
num_files = 5
for i in range(0,num_files):
    folder_name = '/PLDwd_lr_0.001_l1_12_l2_0.235_PLDt_0.5_p_0.6_wd_0.001_idx_{}'.format(i)
    indices = np.load(log_root+folder_name+'/plot_resnonzero_row_idx.npy')
    sequence = np.zeros(784)
    sequence[indices] = 1
    sequences.append(sequence)

var_ratio1 = var_ratio(sequences)
print(var_ratio1)

sequences = np.array(sequences).T


ax[3].spy(sequences.T, aspect=15, markersize=3, origin='lower', c='b')
x_axis1 = ax[3].get_xaxis()

x_axis1.set_ticks_position('bottom')

ax[3].set_xlabel('Row index', fontsize=fontsize)
ax[3].set_xticklabels(range(-100,800,100), fontdict=fontdict)
ax[3].set_yticklabels(range(6,0,-1), fontdict=fontdict)


fig.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

fig.text(0.1, 0.5, 'Training index', va='center', rotation='vertical', fontsize=fontsize)

fig.text(0.13, 0.85, '(a) group-Lasso', va='center', fontsize=fontsize)
fig.text(0.13, 0.8, '15.8x', va='center', fontsize=fontsize)

fig.text(0.13, 0.66, r'(b) GrOWL', va='center', fontsize=fontsize)
fig.text(0.13, 0.61, '16.7X', va='center', fontsize=fontsize)

fig.text(0.13, 0.47, r'(c) group-Lasso+$\ell_2$', va='center', fontsize=fontsize)
fig.text(0.13, 0.42, '23.7X', va='center', fontsize=fontsize)

fig.text(0.13, 0.28, r'(d) GrOWL+$\ell_2$', va='center', fontsize=fontsize)
fig.text(0.13, 0.23, '24.1X', va='center', fontsize=fontsize)
plt.gcf().subplots_adjust(bottom=0.15)

ax[1].xaxis.set_tick_params(width=4)
ax[1].yaxis.set_tick_params(width=4)
ax[0].xaxis.set_tick_params(width=4)
ax[0].yaxis.set_tick_params(width=4)

ax[0].spines['top'].set_linewidth(4)
ax[0].spines['right'].set_linewidth(4)
ax[0].spines['bottom'].set_linewidth(4)
ax[0].spines['left'].set_linewidth(4)

ax[1].spines['top'].set_linewidth(4)
ax[1].spines['right'].set_linewidth(4)
ax[1].spines['bottom'].set_linewidth(4)
ax[1].spines['left'].set_linewidth(4)

ax[3].xaxis.set_tick_params(width=4)
ax[3].yaxis.set_tick_params(width=4)
ax[2].xaxis.set_tick_params(width=4)
ax[2].yaxis.set_tick_params(width=4)

ax[2].spines['top'].set_linewidth(4)
ax[2].spines['right'].set_linewidth(4)
ax[2].spines['bottom'].set_linewidth(4)
ax[2].spines['left'].set_linewidth(4)

ax[3].spines['top'].set_linewidth(4)
ax[3].spines['right'].set_linewidth(4)
ax[3].spines['bottom'].set_linewidth(4)
ax[3].spines['left'].set_linewidth(4)

plt.savefig('mnist_sparsity_pattern_large.jpg')
plt.show()




