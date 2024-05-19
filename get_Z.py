from numba import cuda,float32
import numpy as np
import timeit
import whs as gen
import ptau as pt
from math import comb 
import itertools
from mat_ker import fast_matmul
import time
import gc
from find_id import find_i
from find_res import find_res

@cuda.jit
def add_vec(V1,s):
    tx= cuda.threadIdx.x
    ty= cuda.blockIdx.x
    cuda.atomic.add(V1[ty],tx,s[tx])
    V1[ty][tx]=V1[ty][tx]%2
    
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
        
    

def cutH(H_d,p):
    return H_d[:, :p],H_d[:, p:]



def get_Z(lct,lctt,H0_d,H1_d,V0_d,V1_d,E0_d,E1_d,s_d,comb_tau,comb_t_tau):
    get_E[lct,1](comb_tau,E0_d)
    get_E[lctt,1](comb_t_tau,E1_d)
    cuda.synchronize()
    fast_matmul[(32,32),(32,32)](E0_d,H0_d,V0_d)
    fast_matmul[(32,32),(32,32)](E1_d,H1_d,V1_d)
    add_vec[V1_d.shape[0],V1_d.shape[1]](V1_d,s_d)
    cuda.synchronize()


def main(H,p,tau_p,t,s,ceshi):
    H0, H1 = cutH(H, p)
    s_d = cuda.to_device(np.ascontiguousarray(s))
    
    H0_d = cuda.to_device(np.ascontiguousarray(H0.T))
    H1_d = cuda.to_device(np.ascontiguousarray(H1.T))

    num_p = np.arange(p)
    comb_tau = cuda.to_device(np.array(list(itertools.combinations(num_p, tau_p))))
    num_n_p = np.arange(n - p)
    comb_t_tau = cuda.to_device(np.array(list(itertools.combinations(num_n_p, t - tau_p))))

    lct=len(comb_tau)
    lctt=len(comb_t_tau)

    E0_d = cuda.to_device(np.zeros((lct, p), dtype=np.float32))
    E1_d = cuda.to_device(np.zeros((lctt, n-p), dtype=np.float32))
    V0_d = cuda.to_device(np.zeros((lct, k), dtype=np.float32))
    V1_d = cuda.to_device(np.zeros((lctt, k), dtype=np.float32))
    cuda.synchronize()


    get_Z(lct,lctt,H0_d,H1_d,V0_d,V1_d,E0_d,E1_d,s_d,comb_tau,comb_t_tau)
    V0=np.concatenate((V0_d, np.zeros((lct, 1)), E0_d,np.zeros((lct,n-p))), axis=1)
    V1=np.concatenate((V1_d, np.ones((lctt, 1)), np.zeros((lctt,p)),E1_d), axis=1)
    Z=np.concatenate((V0, V1), axis=0)
    sorted_Z = cuda.to_device(Z[np.lexsort(Z[:, :k].T)])
    result_d = cuda.to_device(np.zeros(Z.shape[0], dtype=np.int32))
    find_i[1,lctt+lct](sorted_Z,k,result_d)
    result_id= np.where(result_d.copy_to_host() == 1)[0]+1
    if result_id.shape[0]!=0:
        block_dim = (32, 32)
        grid_dim = ((result_id.shape[0]*2 + block_dim[0] - 1) // block_dim[0],
                    (Z.shape[1] + block_dim[1] - 1) // block_dim[1])
        find_res[block_dim, grid_dim](sorted_Z,result_id,k,p)
    if result_id[result_id>0].size!=0:
        if np.array_equal(np.dot(H,sorted_Z[result_id[result_id>0][0]-1][k+1:].copy_to_host()),s):
            print("OK:",sorted_Z[result_id[result_id>0][0]-1][k+1:].copy_to_host()) 
    
        
        #return sorted_Z[result_id[result_id>0][0]-1][k+1:].copy_to_host()

start=time.time()
n = 10
seed = 1
w, H, s = gen.get_data(n, seed)
k,n=H.shape
#print(H,s)
ceshi=np.zeros(n)
start=time.time()
for t in range(w-1):
    P_tau = pt.get_P_tau(n, t)
    for p,tau_p in P_tau:
        #print(t,p,tau_p)
        main(H,p,tau_p,t,s,ceshi)
        #print("ceshi:",ceshi)
        
print(time.time()-start)

        





