import numpy as np
from numba import cuda
import itertools
import time
from numba import jit
@cuda.jit
def generate_num_p(num_p):
    i = cuda.grid(1)
    if i < num_p.shape[0]:
        num_p[i] = i

@cuda.jit
def generate_combinations(num_p, comb_tau):
    i = cuda.grid(1)
    if i < comb_tau.shape[0]:
        n = num_p.shape[0]
        r = 2
        index = 0
        for j in range(r):
            while index < n and i >= (n - index - 1):
                i -= (n - index - 1)
                index += 1
            comb_tau[i, j] = num_p[index]
            index += 1

def get_num_p(n,num_p):
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    generate_num_p[blocks_per_grid, threads_per_block](num_p)
    
def get_num_comb(num_p, comb_tau):
    threads_per_block = 256
    blocks_per_grid = (comb_tau.shape[0] + threads_per_block - 1) // threads_per_block
    generate_combinations[blocks_per_grid, threads_per_block](num_p, comb_tau)

n = 100
p=3
num_p = cuda.device_array(n, dtype=np.int32)
get_num_p(n,num_p)

print(num_p.copy_to_host())