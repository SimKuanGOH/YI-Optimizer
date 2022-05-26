import numpy as np
from yi import yialgo
#from cec2017.functions import all_functions
from cec17_functions import cec17_test_func
from mpi4py import MPI

def func(x, nx, fn):
    f = [0]
    cec17_test_func(x, f, nx=nx, mx=1, func_num=fn)
    return f[0]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpisize = comm.Get_size()
d=50
lims = np.ones((2, d))
lims[0,:] = -100
lims[1,:] = 100
itmax = 10000*d
fitness = np.zeros((30))
sigma=d
alpha=3
for i in range(30):
    p1, f1,curves = yialgo(lambda x:func(x, nx=d, fn=i+1), itmax, lbounds = lims[0,:], ubounds = lims[1,:],
                    Imin = 6,Imax = 15,no_copy=2*d, d = d, sigma=sigma, adapt=True, alpha=alpha,reverse=True)
    fitness[i] = f1
    
    
    np.savetxt('./dat/fitness_yi_s'+str(d)+'a'+str(alpha)+'_re_D'+str(d)+'_run'+str(rank)+"_Prob_"+str(i)+'_curves.csv', np.array(curves), delimiter=',')
    print('rank:',rank,' func:',i,' finished')

np.save('./dat/fitness_yi_s'+str(sigma)+'a'+str(alpha)+'_re_D'+str(d)+'_run'+str(rank)+'.npy', fitness)
    
