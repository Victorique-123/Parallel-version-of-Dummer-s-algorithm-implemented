import numpy as np
from numba import cuda
import time
@cuda.jit
def matrix_multiply_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        sum = 0
        for k in range(A.shape[1]):
            sum += A[row, k] * B[k, col]
        C[row, col] = sum%2

def matrix_multiply(A, B, C):
    assert A.shape[1] == B.shape[0], "Incompatible matrix sizes"
    if A.size != 0 and B.size != 0:
        
        block_dim = (32, 32)
        grid_dim = ((C.shape[0] + block_dim[0] - 1) // block_dim[0],
                    (C.shape[1] + block_dim[1] - 1) // block_dim[1])
        matrix_multiply_kernel[grid_dim, block_dim](A, B, C)