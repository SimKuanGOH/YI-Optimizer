import numpy as np
from scipy.optimize import differential_evolution
# from cec2017.functions import all_functions
from sklearn import preprocessing
import random
import math
import pandas as pd

def levy_flight(Lambda,size=2,sigma2=1):
    #generate step from levy distribution
    sigma1 = np.power((math.gamma(1 + Lambda) * np.sin((np.pi * Lambda) / 2)) \
                      / math.gamma((1 + Lambda) / 2) * np.power(2, (Lambda - 1) / 2), 1 / Lambda)
#     sigma2 = 1
    u = np.random.normal(0, sigma1, size=size)
    v = np.random.normal(0, sigma2, size=size)
    step = u / np.power(np.fabs(v), 1 / Lambda)

    return step    # return np.array (ex. [ 1.37861233 -1.49481199  1.38124823])

def split_yi(fun, p, lbounds, ubounds, fun_it, no_copy, sigma=40, Lambda=1.5):
    """
    fun_it: function iteration
    """
    dims = len(p)
    pold = p.copy()
    for i in range(no_copy):
        pnew = pold + sigma*levy_flight(Lambda,sigma2=1, size=len(p))
        pnew[pnew<lbounds] = lbounds[pnew<lbounds] + np.random.rand((np.sum(pnew<lbounds)))*(ubounds[pnew<lbounds]-lbounds[pnew<lbounds])
        pnew[pnew>ubounds] = lbounds[pnew>ubounds] + np.random.rand((np.sum(pnew>ubounds)))*(ubounds[pnew>ubounds]-lbounds[pnew>ubounds])
        if i==0:
            f = fun(pnew)
            fun_it = fun_it+1
        else:
            fnew = fun(pnew)
            fun_it = fun_it+1
            if fnew<f:
                f = fnew
                p = pnew.copy()
    return p, f, fun_it

def yialgo(fun, max_fun, lbounds = np.array([0, 0]), ubounds = np.array([5, 5]), 
            Imin = 5,Imax = 10, d = 2, sigma=40, no_copy=4, adapt=False, alpha=5,reverse=True,
           adapt_l=False, Lambda=1.5, alpha_l=1.1):
    
    curves = []
    
    pm = (d/(d+5))*(d/(d+5))
    pu = 1 - pm
    
    p1 = lbounds + (np.random.rand(d))*(ubounds-lbounds)

    f1 = fun(p1)
    fun_it = 1 

    p_arch = []
    i = 0
    j = 0
    fe_ = np.arange(max_fun/(Imax-Imin+1),max_fun+10,max_fun/(Imax-Imin+1))
    Isel = np.arange(Imin, Imax+1)
    if reverse:
        Isel = Isel[::-1]
    fe = fe_[0]
    I_ = Isel[0]
    while (True):
        if fun_it>fe:
            j=j+1
            fe = fe_[j]
            I_ = Isel[j]
            if adapt:
                sigma = sigma/alpha
            if adapt_l:
                Lambda = Lambda/alpha_l
            i = 0
        p_arch.append([p1, f1])
        i = i+1
        if (fun_it>max_fun-no_copy):
            no_copy=max_fun-fun_it
        p1, f1, fun_it= split_yi(fun, p1, lbounds, ubounds,fun_it,no_copy=no_copy, sigma=sigma, Lambda=Lambda)
        
        curves.append(f1)

        if (i==I_) or (fun_it>=max_fun):
            p_arch.append([p1, f1])
            p_arch = np.array(p_arch,dtype=object)
            ind = np.argsort(p_arch[:,1])
            p1, f1 = p_arch[ind[0], :]
            p_arch = []
            i = 0
            
            if (fun_it>=max_fun) or (j==Imax-Imin+1):
                return p1, f1, curves 
