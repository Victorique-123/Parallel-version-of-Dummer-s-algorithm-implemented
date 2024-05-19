import random
import math
import numpy as np

def dGV(n, k):
    d = 0
    aux = 2**(n-k)
    b = 1
    while aux >= 0:
        aux -= b
        d += 1
        b *= (n-d+1)
        b /= d
    return d 
    
def recover_matrices(half_ht):
    n = 2 * len(half_ht)
    k = n // 2
    # Construct the complete check matrix H
    h = np.hstack((half_ht.T, np.eye(k, dtype=int)))
    return h
    
def get_data(n, seed):
    w = math.ceil(1.05 * dGV(n,n//2))
    random.seed(seed)
    n = n
    seed = seed
    half_ht=[]
    line = ""
    for i in range(n-n//2):
        for j in range(n//2):
            line += str(random.randint(0,1))
        arr = [int(c) for c in line]
        line = ""
        half_ht.append(arr)
    line = ""

    for j in range(n//2):
        line += str(random.randint(0,1))
    s = [int(c) for c in line]
    H=recover_matrices(np.array(half_ht))
    return w,H,s


