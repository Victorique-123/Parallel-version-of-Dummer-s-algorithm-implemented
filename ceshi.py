import numpy as np
from numba import cuda
import math


@cuda.jit
def generate_arange_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        arr[idx] = idx
        



@cuda.jit
def generate_combinations_kernel(arr, r, out):
    idx = cuda.grid(1)
    if idx < out.shape[0]:
        start = idx
        curr = 0
        for i in range(arr.shape[0]):
            if curr < r:
                out[idx, curr] = arr[i]
                curr += 1
            else:
                if start < arr.shape[0]:
                    i = start
                    start += 1
                    curr = 0
                    out[idx, curr] = arr[i]
                    curr += 1

def generate_arange_gpu(p):
    arr = cuda.device_array(p, dtype=np.int32)
    
    threads_per_block = 256
    blocks_per_grid = (p + threads_per_block - 1) // threads_per_block
    
    generate_arange_kernel[blocks_per_grid, threads_per_block](arr)
    
    return arr

def generate_combinations(arr, r):
    n = arr.shape[0]
    num_combinations = math.comb(n, r)
    out = cuda.device_array((num_combinations, r), dtype=arr.dtype)
    
    threads_per_block = 256
    blocks_per_grid = (num_combinations + threads_per_block - 1) // threads_per_block
    generate_combinations_kernel[blocks_per_grid, threads_per_block](arr, r, out)
    return out.copy_to_host()


p=3
n=10
t=2
tau_p=1
# 使用示例
num_p = generate_arange_gpu(p)
comb_tau = generate_combinations(num_p, tau_p)

num_n_p = generate_arange_gpu(n-p)
comb_t_tau = generate_combinations(num_n_p, t - tau_p)

print(num_p.copy_to_host())
print(num_n_p.copy_to_host())

print(comb_tau)
print(comb_t_tau)