import numpy as np

def reshape_2D_4D(X, target_shape, reshape_type, reshape_order, config):
    """
    This function transform the 2D tensor to 4D tensor or vice versa. 

    Args:
        X: input tensor, either 2D or 4D numpy array
        target_shape: target shape, must be provided if reshape_type=1
        reshape_type: 
                   1: 2D to 4D
                   2: 4D to 2D
        reshape_order: "C" or "F", please refer to the arg "order" in numpy.reshape 
        config: the yaml configuration file

    Returns:
        Y: output tensor, either 2D or 4D numpy array
    """

    param_shape = X.shape

    #Input tensor must be either 2D or 4D
    if len(param_shape) != 2 and len(param_shape) != 4:
        raise Exception("The input tensor for the reshape_2D_4D much be either 2D or 4D!")

    #Reshape 2D tensor to 4D
    if reshape_type == 1:
        #First transpose the 2D input tensor
        if config['conv_reg_type'] != 4:
            X_T = np.transpose(X)
        else:
            X_T = X
        
        if (config['conv_reg_type'] == 1) or (config['conv_reg_type'] == 3) or (config['conv_reg_type'] == 4):
            Y = np.reshape(X_T, target_shape, order=reshape_order)
        elif config['conv_reg_type'] == 2:
            #the target the shape should be the swapped version of the original one
            target_shape_swap = list(target_shape)
            target_shape_swap[2], target_shape_swap[3] = target_shape_swap[3], target_shape_swap[2]
            Y_swap = np.reshape(X_T, tuple(target_shape_swap), order=reshape_order)
            Y = np.swapaxes(Y_swap, 2,3)
        else:
            raise Exception('Please specify the regularization type for convolutional layers')
                

    elif reshape_type == 2:
        #prune output channels
        if config['conv_reg_type'] == 1:
            Y_T = np.reshape(X, (np.prod(param_shape[0:3]), param_shape[3]), order=reshape_order)

        #prune input depth
        elif config['conv_reg_type'] == 2:
            X_swapaxes = np.swapaxes(X, 2,3)
            param_swap_shape = X_swapaxes.shape
            Y_T = np.reshape(X_swapaxes, (np.prod(param_swap_shape[0:3]), param_swap_shape[3]), order=reshape_order)

        #prune filter
        elif config['conv_reg_type'] == 3:
            Y_T = np.reshape(X, (np.prod(param_shape[0:2]), np.prod(param_shape[2:])), order=reshape_order)
        
        #arbitrary filter
        elif config['conv_reg_type'] == 4:
            Y = np.reshape(X, (np.prod(param_shape[0:3]), param_shape[3]), order=reshape_order)

        
        else:
            raise Exception('Please specify the regularization type for convolutional layers')

        #Transpose the reshape matrix in order to call the row-wise prox_function 
        if config['conv_reg_type'] != 4:
            Y = np.transpose(Y_T)


    return Y