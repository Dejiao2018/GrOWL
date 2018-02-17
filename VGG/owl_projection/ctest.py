import ctypes 
from ctypes import c_size_t, c_double, c_int, POINTER, pointer, c_float
import numpy as np
from numpy.random import randn
import os


# _proxLib = ctypes.CDLL(os.path.dirname(__file__) + "libprox.so")
_proxLib = ctypes.CDLL(os.path.dirname(__file__) + "/libprox.so")

_proxLib.evaluateProx.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, POINTER(c_int)]
_proxLib.evaluateProx.restype = POINTER(c_float)
_proxLib.evaluateProx.argtypes = [POINTER(c_float)]

def __init__():
    pass

    
def proxOWL_part(y,omega):
    global _proxLib
    numy = y.size
    array_type = c_float * numy
    null_ptr = POINTER(c_int)()

    # call C function
    r = _proxLib.evaluateProx(array_type(*y),array_type(*omega),c_size_t(numy),null_ptr)

    # transform the returned pointer to array
    v = np.array(np.fromiter(r, dtype = np.float64, count = numy))
    
    # free the memory of r   
    _proxLib.free_mem(r)
    return v

