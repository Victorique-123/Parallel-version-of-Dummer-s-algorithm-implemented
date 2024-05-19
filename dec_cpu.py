
import numpy as np
from itertools import combinations
from scipy.special import comb
from random import choice
from whs import get_data
import timeit
import itertools
import ptau as pt
import time


def binary_to_int(binary_row):
    binary_str = ''.join(str(bit) for bit in binary_row[:3])
    return int(binary_str, 2)

def binary_addition(a, b):
    return [ai ^ bi for ai, bi in zip(a, b)]

def process_array(array, n, k):
    array = np.array(array)
    result = set()
    i = 0
    
    array = array[np.lexsort(array[:, :n - k + 1].T[::-1])]
    
    while i < len(array):
        temp = []
        group_start = i

        while i + 1 < len(array) and (array[i, :n - k] == array[i + 1, :n - k]).all():
            i += 1
        group_end = i
        
        for r1 in range(group_start, group_end + 1):
          for r2 in range(r1, group_end + 1):
            if array[r1, n - k] == 0 and array[r2, n - k] == 1:
              addition_result = binary_addition(array[r1, -n:], array[r2, -n:])
              result.add(tuple(addition_result))
        i += 1

    return np.array([list(r) for r in result])


def find_e(s, w, H): # s- синдром, w-максимальное количество ошибок, H-проверочная матрица
  k,n = H.shape
  c_full=[]
  print(w)
  for t in range(1, w + 1):
      p_and_tau = pt.get_P_tau(n, t)
      #print(len(p_and_tau))
      for x in p_and_tau:
        p, tau_p = x
        H0 = H[:, :p]
        H1 = H[:, p:]
        num_p = list(range(p))
        comb_tau = list(itertools.combinations(num_p,tau_p))

        num_n_p = list(range(n-p))
        comb_t_tau = list(itertools.combinations(num_n_p, t-tau_p))

        Z0 = []
        for i in comb_tau:
          e_0 = np.zeros(p)
          e_0[list(i)] = 1
          v0 = np.mod(np.dot(e_0, H0.T),2).astype(int)
          ax = np.concatenate((v0, [0], e_0, np.zeros(n-p)))
          Z0.append(ax)
          
          #print("ceshi",np.mod(np.dot(H0,e_0),2).astype(int))

        Z1 = []
        for i in comb_t_tau:
          e_1 = np.zeros(n-p)
          e_1[list(i)]=1
          v1=np.mod(np.dot(e_1, H1.T),2).astype(int)
          bx=np.concatenate((np.mod(v1+s,2), [1], np.zeros(p),e_1))
          Z1.append(bx)
          
        Z = np.array(Z0 + Z1).astype(int)
        start=time.time()
        Z = np.array(sorted(Z, key=binary_to_int)).astype(int)
        #print("shijian:",time.time()-start)
        box=process_array(Z, n, k)
        if len(box)!=0:
          #print(H0)
          #print(t,p,tau_p,t)
          #print(num_p,comb_tau,num_n_p,comb_t_tau)
          #print(Z)

          return box
  return c_full
n=40
seed=1
w, H, s = get_data(n, seed)
print("s:",s)
#print(H)
#print(s)
execution_time = timeit.timeit(lambda: print(find_e(s, w, H)), number=1)
print(f"外函数执行时间: {execution_time:.6f} 秒")

