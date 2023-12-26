import numpy as np
import time
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

def gpu_add(a, b):
    a_gpu = drv.mem_alloc(a.nbytes)
    b_gpu = drv.mem_alloc(b.nbytes)
    result_gpu = drv.mem_alloc(a.nbytes)

    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)

    codeK = SourceModule("""
    __global__ void add_vectors(int *a, int *b, int *result)
    {
        const int idx = threadIdx.x + blockDim.x * blockIdx.x;
        result[idx] = a[idx] + b[idx];
    }
    """)

    func = codeK.get_function("add_vectors")
    start = time()
    func(a_gpu, b_gpu, result_gpu, block=(1024,1,1), grid=(n//1024+1,1))
    gpu_time = time() - start

    result = np.empty_like(a)
    drv.memcpy_dtoh(result, result_gpu)

    a_gpu.free()
    b_gpu.free()
    result_gpu.free()

    return result, gpu_time

def cpu_add(a, b):
    start = time()
    result = a + b
    cpu_time = time() - start
    return result, cpu_time

# 创建较大的二进制向量
n = 700000000
a = np.random.randint(2, size=n).astype(np.int32)
b = np.random.randint(2, size=n).astype(np.int32)

# 比较
try:
    gpu_result, gpu_time = gpu_add(a, b)
    print(f"GPU Execution Time: {gpu_time} seconds")
except NameError:
    print("GPU function is not defined due to missing PyCUDA.")

cpu_result, cpu_time = cpu_add(a, b)
print(f"CPU Execution Time: {cpu_time} seconds")