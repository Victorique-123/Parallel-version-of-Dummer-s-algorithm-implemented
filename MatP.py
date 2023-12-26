import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
from time import time

# CUDA C代码
kernel_code = """
__global__ void MatrixMulKernel(float *a, float *b, float *c, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < width && col < width)
    {
        float result = 0;
        for (int k = 0; k < width; ++k)
        {
            result += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = result;
    }
}
"""

# 编译内核代码
codeK = SourceModule(kernel_code)

# 创建矩阵乘法内核函数
matmul = codeK.get_function("MatrixMulKernel")

# 生成随机矩阵
size = 5000 # 矩阵大小
a = np.random.rand(size, size).astype(np.float32)
b = np.random.rand(size, size).astype(np.float32)

# 输出矩阵
c = np.zeros((size, size), dtype=np.float32)

n=16
# 定义线程块的大小和网格的大小
block_size = (n, n, 1) # 线程块大小
grid_size = (int(np.ceil(size / block_size[0])), int(np.ceil(size / block_size[1])))

# 在GPU上执行矩阵乘法
start=time()
matmul(
    drv.In(a), drv.In(b), drv.Out(c),
    np.int32(size),
    block=block_size, grid=grid_size
)
print("GPU", time()-start)


start = time()
c = np.dot(a, b)
print("CPU", time() - start)