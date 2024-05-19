import numpy as np
from scipy.special import comb
from numba import cuda
from math import comb
def get_P_tau(n, t):
    p_and_tau = []
    for tau_p in range(t):
      for p in range(n-1):      
        if (comb(p+1,(tau_p+1))>comb(n-p-1,t-(tau_p+1))) and (comb(p,tau_p)<=comb(n-p,t-tau_p)):
          p_and_tau.append((p,tau_p))
    p_and_tau=list(set(p_and_tau))
    return p_and_tau
  
  