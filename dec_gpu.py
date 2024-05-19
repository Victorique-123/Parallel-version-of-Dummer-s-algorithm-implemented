from numba import cuda,float32
import numpy as np
import whs as gen
import ptau as pt
import itertools
from ceshi3 import matrix_multiply
import time
from find_id import find_i
from find_res import find_res
import os
import time
import base as base
from numba import cuda,config


os.environ['CUDA_WARN_ON_IMPLICIT_COPY'] = '0'
cuda.current_context().log_level = 0
config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
config.CUDA_WARN_ON_IMPLICIT_COPY = 0



@cuda.jit
def add_vec(V1,s):
    tx= cuda.threadIdx.x
    ty= cuda.blockIdx.x
    cuda.atomic.add(V1[ty],tx,s[tx])
    V1[ty][tx]=V1[ty][tx]%2
    
@cuda.jit
def get_E_new(comb_tau,E):
    x,y=cuda.grid(2)
    if x<comb_tau.shape[0] and y<comb_tau.shape[1]:
        E[x,int(comb_tau[x][y])]=1   
    

@cuda.jit
def get_E(comb_tau,E):
    ty=cuda.blockIdx.x
    tx=cuda.threadIdx.x
    for i in comb_tau[ty]:
        E[ty, int(i)] = 1


@cuda.jit
def getE2(comb_tau,E):
    tx = cuda.threadIdx.x # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # Similarly, this is the unique block ID within the 1D grid
    E[ty][comb_tau[ty][tx]]=1
        
@cuda.jit
def initialize_array(arr):
    i, j = cuda.grid(2)
    if i < arr.shape[0] and j < arr.shape[1]:
        arr[i, j] = 0



def cutH(H_d,p):
    return H_d[:, :p],H_d[:, p:]

def get_Z_new(H0_d,H1_d,V0_d,V1_d,E0_d,E1_d,s_d,comb_tau,comb_t_tau):
    if comb_tau.size!=0:
        block_dimE0 = (1024,1024)
        grid_dimE0 = ((comb_tau.shape[0] + block_dimE0[0] - 1) // block_dimE0[0],
                    (comb_tau.shape[1] + block_dimE0[1] - 1) // block_dimE0[1],)
        get_E_new[block_dimE0,grid_dimE0](comb_tau,E0_d)
        #get_E[block_dim,grid_dim](comb_t_tau,E1_d)
    cuda.synchronize()
    if comb_t_tau.size!=0:
        block_dimE1 = (1024,1024)
        grid_dimE1 = ((comb_t_tau.shape[0] + block_dimE1[0] - 1) // block_dimE1[0],
                    (comb_t_tau.shape[1] + block_dimE1[1] - 1) // block_dimE1[1],)
        get_E_new[block_dimE1,grid_dimE1](comb_t_tau,E1_d)
    cuda.synchronize()
    #修改这里
    matrix_multiply(E0_d,H0_d,V0_d)
    matrix_multiply(E1_d,H1_d,V1_d)
    add_vec[V1_d.shape[0],V1_d.shape[1]](V1_d,s_d)
    cuda.synchronize()

def get_Z_new_2(H0_d,H1_d,V0_d,V1_d,E0_d,E1_d,s_d,comb_tau,comb_t_tau):
    if comb_tau.size!=0:
        block_dimE0 = (16, 16)
        grid_dimE0 = ((comb_tau.shape[0] + block_dimE0[0] - 1) // block_dimE0[0],
                    (comb_tau.shape[1] + block_dimE0[1] - 1) // block_dimE0[1],)
        get_E_new[block_dimE0,grid_dimE0](comb_tau,E0_d)
        #get_E[block_dim,grid_dim](comb_t_tau,E1_d)
    cuda.synchronize()
    if comb_t_tau.size!=0:
        block_dimE1 = (16, 16)
        grid_dimE1 = ((comb_t_tau.shape[0] + block_dimE1[0] - 1) // block_dimE1[0],
                    (comb_t_tau.shape[1] + block_dimE1[1] - 1) // block_dimE1[1],)
        get_E_new[block_dimE1,grid_dimE1](comb_t_tau,E1_d)
    cuda.synchronize()
    #修改这里
    matrix_multiply(E0_d,H0_d,V0_d)
    matrix_multiply(E1_d,H1_d,V1_d)
    add_vec[V1_d.shape[0],V1_d.shape[1]](V1_d,s_d)
    cuda.synchronize()


def main(H,p,tau_p,t,s):
    start0=time.time()
    
    s_d = cuda.to_device(np.ascontiguousarray(s))
    H0_d=cuda.device_array((p,H.shape[0]))
    H1_d=cuda.device_array((n-p,H.shape[0]))
    
    T1=time.time()
    base.cut0(H.T,H0_d,p)
    base.cut1(H.T,H1_d,p)
    

    num_p = np.arange(p)
    comb_tau = cuda.to_device(np.array(list(itertools.combinations(num_p, tau_p))))
    num_n_p = np.arange(n - p)
    comb_t_tau = cuda.to_device(np.array(list(itertools.combinations(num_n_p, t - tau_p))))
    

    lct=len(comb_tau)
    lctt=len(comb_t_tau)
    T1e=time.time()-T1
    """
    #start=time.time()
    E0_d = cuda.to_device(np.zeros((lct, p), dtype=np.int32))
    E1_d = cuda.to_device(np.zeros((lctt, n-p), dtype=np.int32))
    V0_d= cuda.device_array((lct, k), dtype=np.int32)
    V1_d= cuda.device_array((lctt, k), dtype=np.int32)
    #print(time.time()-start)
    """
    
    E0_d= cuda.device_array((lct, p), dtype=np.int32)
    E1_d= cuda.device_array((lctt, n-p), dtype=np.int32)
    V0_d= cuda.device_array((lct, k), dtype=np.int32)
    V1_d= cuda.device_array((lctt, k), dtype=np.int32)
    
    threads_per_block = (16, 16)
    blocks_per_grid = ((E0_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0], 
                       (E0_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])
    initialize_array[blocks_per_grid, threads_per_block](E0_d)
    threads_per_block = (16, 16)
    blocks_per_grid = ((E1_d.shape[0] + threads_per_block[0] - 1) // threads_per_block[0], 
                       (E1_d.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])
    initialize_array[blocks_per_grid, threads_per_block](E1_d)


    cuda.synchronize()
    #print("T0:",time.time()-start0)
    
    T2=time.time()
    get_Z_new(H0_d,H1_d,V0_d,V1_d,E0_d,E1_d,s_d,comb_tau,comb_t_tau)
    cuda.synchronize()
    V0=np.concatenate((V0_d, np.zeros((lct, 1)), E0_d,np.zeros((lct,n-p))), axis=1)
    V1=np.concatenate((V1_d, np.ones((lctt, 1)), np.zeros((lctt,p)),E1_d), axis=1)

    Z=np.concatenate((V0, V1), axis=0)
    sorted_Z = cuda.to_device(Z[np.lexsort(Z[:, :k].T)])
    result_d = cuda.to_device(np.zeros(Z.shape[0], dtype=np.int32))
    T2e=time.time()-T2
    #print(result_d.copy_to_host())
    
    block_dim=1024
    grid_dim=(Z.shape[0] + block_dim - 1) // block_dim
    #print(block_dim,grid_dim)
    
    T3=time.time()
    find_i[block_dim,grid_dim](sorted_Z,k,result_d)
    
    
    result_id= np.where(result_d.copy_to_host() == 1)[0]+1
    #print(sorted_Z.copy_to_host())
    if result_id.shape[0]!=0:
        block_dim = (64, 64)
        grid_dim = ((result_id.shape[0]*2 + block_dim[0] - 1) // block_dim[0],
                    (Z.shape[1] + block_dim[1] - 1) // block_dim[1])
        find_res[block_dim, grid_dim](sorted_Z,result_id,k,p)
    #print("T1",time.time()-start)
    T3e=time.time()-T3
    if result_id[result_id>0].size!=0:
        print(sorted_Z[result_id[result_id>0][0]-1][k+1:].copy_to_host())
        return 1,T1e+T2e+T3e
    else:
        return 0,T1e+T2e+T3e

    
n =70
seed = 1
w, H, s = gen.get_data(n, seed)
k,n=H.shape
res=0
T=0
start=time.time()
for t in range(w-1):
    P_tau = pt.get_P_tau(n, t)
    for p,tau_p in P_tau:
        res,time_block=main(H,p,tau_p,t,s)
        T=T+time_block
        if res==1:
            print("OK")
            break
    if res==1:
        break
print(time.time()-start)
print("time dec:",T)
