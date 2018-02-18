from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

'''
Code for generating figure 6

Please modify the filenames accordingly 
'''

folder_root = ['./GrOWL', './GrOWL_L2', './GrLasso', './GrLasso_L2', './wd']
labels = ['GrOWL', r'GrOWL+$\ell_2$', "group-Lasso",r"group-Lasso+$\ell_2$",'weight decay'    ]
res_dir = '/res_dir'
filenames = ['/layer2_similarity_150.npy', '/layer6_similarity_150.npy', '/layer10_similarity_150.npy', '/layer11_similarity_150.npy']
title = ['Layer 1', 'Layer 6', 'Layer 10', 'Layer 11']

font = {"size" : 30}

fig = plt.figure(figsize=(40,8))
for fig_idx, file in enumerate(filenames):
    ax = fig.add_subplot(1,4,fig_idx+1, projection='3d')
    for hist_idx, folder in enumerate(folder_root):
        sim = np.reshape(np.load(folder+'/res_dir'+file), (-1, 1))
        hist, bins = np.histogram(sim/np.max(sim), range=[0.1, 1.2], density=True, bins='auto')
        xs = (bins[:-1] + bins[1:])/2
        ax.bar(xs, hist, zs=1 * (hist_idx+1), zdir='y', alpha=0.8,width=0.05)
    ax.set_xlim([-0.15,1.15])
    ax.set_xticks([0,0.5,1])
    ax.set_xlabel('Similarity', labelpad=25)
    ax.set_yticks([1,2,3,4,5])
    ax.set_yticklabels(['GO','GOL','GL','GLL','WD'], rotation=-25)
    ax.set_zlabel('Density', labelpad=20)
    ax.set_title(title[fig_idx])
matplotlib.rc('font', **font)
plt.tight_layout()
plt.savefig('histogram.png')