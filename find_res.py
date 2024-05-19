from numba import cuda
import numpy as np


@cuda.jit
def find_res(Z,result_id,k,p):
    x, y = cuda.grid(2)
    if x < result_id.shape[0] and y < Z.shape[1]:
        is_equal = True
        for i in range(k):
            if Z[result_id[x]-1, i] != Z[result_id[x], i]:
                is_equal = False
                break
        if is_equal:
            if y>=p+k+1:
                Z[result_id[x]-1, y]=Z[result_id[x],y]
        else:
            result_id[x]=-1
            
