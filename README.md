# GrOWL
## Learning to share: simultaneously parameter tying and sparsification in deep learning

Dejiao Zhang*, Haozhu Wang*, Mario Figueiredo, Laura Balzano (*Co-first author)  
https://openreview.net/pdf?id=rypT3fb0b

## To run the code:
### 1. Complie the the c code in ./owl_projection by the following:
   gcc -fPIC -shared -o libprox.so proxSortedL1.c

### 2. VGG on Cifar10 (VGG folder)
    python vgg_main.py 
    
    You can switch to different regularizers by changing the configuration info in "flags.py"  
    All configuration settings are in vgg_main.py and flag.py  
    Some of the hyperparameters are described in the table, e.g., preference value

### 3. Fully connected network on MNIST (MNIST folder)
    python run_exp.py (reproduce results in table 1)
    
    All available configuration settings are contained in experiment_config folders,   
    you can modify the settings either in the .yaml file "10_20_config_search.yaml" or in the "run_exp.py" script.

### 4. Plots (Plots folder)
    python gen_fig4.py (generate figure 4)
    python gen_fig5.py (generate figure 5)
    python gen_fig6.py (generate figure 6)

    To run the the above codes successfully, please modify the "file_names" and "log_root"   
    in the code to match your local files. 

## Dependencies:
Tensorflow 1.0.0  
Numpy 1.14.0  
Scipy 1.0.0  
Matplotlib 2.1.0  
Scikit-learn 0.19.1  
