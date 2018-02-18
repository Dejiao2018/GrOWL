# GrOWL
## Learning to share: simultaneously parameter tying and sparsification in deep learning

Dejiao Zhang*, Haozhu Wang*, Mario Figueiredo, Laura Balzano (*Co-first author)

## To run the code:
### 1. Complie the the c code in ./owl_projection by the following:
   gcc -fPIC -shared -o libprox.so proxSortedL1.c

### 2. VGG on Cifar10 
    python vgg_main.py (reproduce table 2)
    python run_expr.py (reproduce table 4)
    
    You can switch to different regularizers by changing the configuration info in flags.py
    
## Dependencies:
Tensorflow 1.0.0
Numpy   
Scipy  
Matplotlib  
Scikit-learn  
