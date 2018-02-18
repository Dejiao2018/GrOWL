import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''
Code for generating figure 4
'''

filenames = ['correlation.npy', 'growl.npy', 'growl_l2.npy', 'group_lasso.npy','group_lasso_l2.npy', 'weight_decay_0.05.npy', ]
titles = ['(a) MNIST correlation', '(b) GrOWL', r'(c) GrOWL + $\ell_2$', '(d) Group Lasso', r'(e) Group Lasso + $\ell_2$', '(f) Weight decay']
nonzero_ratio = [80.4, 83.60, 87.6, 93.2, 0]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,6))
axes = np.reshape(axes, (-1, ))
print(axes)
for idx, filename in enumerate(filenames):
    correlation = np.load(filename)
    num_row = np.shape(correlation)[0]
    ax = axes[idx]
    im = ax.imshow(correlation, vmin=-1, vmax=1, cmap='seismic')
    ax.set_title(titles[idx])
    ax.set_xticks([0,100,300,500,700])
    ax.set_yticks([0,100,300,500,700])
    if idx >= 1:
    	ax.set_xlabel('Sparsity {}%'.format(nonzero_ratio[idx-1]))
    ax.set_xlim([0,784])
    ax.set_ylim([0,784])
    ax.tick_params(direction='in')

fig.tight_layout()
fig.subplots_adjust(right=0.80)
cbar_ax = fig.add_axes([0.78, 0.09, 0.01, 0.83])
fig.colorbar(im, cax=cbar_ax, ticks=[-1,0,1])
plt.savefig('similarity.png')

plt.show()