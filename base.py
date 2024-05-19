from numba import cuda
import numpy as np
import whs as gen
@cuda.jit
def cut0_d(H,H0,p):
    x,y=cuda.grid(2)
    if x<H0.shape[0] and y<H0.shape[1]:
        H0[x,y]=H[x,y]

@cuda.jit
def cut1_d(H,H1,p):
    x,y=cuda.grid(2)
    if x<H1.shape[0] and y<H1.shape[1]:
        H1[x,y]=H[x+p,y]
        
        

@cuda.jit
def get_comb_d():
    pass


def cut0(H,H0,p):
    Block_dim=(32,32)
    grid_dim=(H0.shape[0]+Block_dim[0]+1//Block_dim[0],
              H0.shape[1]+Block_dim[1]+1//Block_dim[1])
    cut0_d[grid_dim, Block_dim](H,H0,p)

def cut1(H,H1,p):
    Block_dim=(32,32)
    grid_dim=(H1.shape[0]+Block_dim[0]+1//Block_dim[0],
              H1.shape[1]+Block_dim[1]+1//Block_dim[1])
    cut1_d[grid_dim, Block_dim](H,H1,p)


def get_comb(p,tau_p):
    pass

"""n =10
seed = 1
w, H, s = gen.get_data(n, seed)
k,n=H.shape
res=0
p=3
   
H0=cuda.device_array((p,H.shape[0]))
H1=cuda.device_array((n-p,H.shape[0]))
cut0(H.T,H0,p)
cut1(H.T,H1,p)
print(H)
print(H0.copy_to_host())
print(H1.copy_to_host())"""






