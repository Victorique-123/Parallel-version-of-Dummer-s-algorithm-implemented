from numba import cuda
import numpy as np


@cuda.jit
def find_i(Z,k,result):
    x=cuda.grid(1)
    if x<Z.shape[0]-1:
        if Z[x,k]==0 and Z[x+1,k]==1:
            result[x]=1



