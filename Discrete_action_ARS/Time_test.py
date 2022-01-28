#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 10:58:02 2021

@author: lukas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:21:25 2021

@author: lukas
"""

import numpy as np
#import benchmark_class
import ARS_benchmark

import matplotlib.pyplot as plt
import time
#from sklearn.manifold import TSNE
N = 20

noise = 0.0 # remember noise is optional

Tlist = []
# Note, if the noise is different from zero, you get a different result each time. 
pulse = np.ones(N)*0.5#np.random.rand(N)
alpha = 0.12
v = 0.25
alpha2 = 0.1
v2 = 0.4
maxite = 20
maxepochs = 100
f_story = np.zeros((maxepochs,maxite))
times = np.zeros((maxepochs,maxite))
M_list = np.zeros((N,maxite))
F_list = np.zeros((100,maxite))
ARS = ARS_benchmark.ARSTrainer()

T = 3+1/3

s_list = []
l = None
tim = []
for i in range(maxite):
    pulse = np.random.rand(N)
    #M_list[:,i],s,f_story[:,i],times[:,i] = ARS.train(pulse,N,T,alpha,v,L=l, Noise = noise,maxepochs=maxepochs)
    t0 = time.time()
    _,_,ac,_ = ARS.train(pulse,N,T,alpha,v,L=l, Noise = noise,maxepochs=maxepochs)
    print(ac)
    t1 = time.time()
    pulse = np.random.rand(N)
    #l += 1
    tim.append(t1-t0)
    #s_list.append(s)



time_average_A = np.average(times,axis = 1)
f_story_avg_A = np.average(f_story,axis = 1)
f_story_std_A = np.std(f_story,axis = 1)
f_story_max_A = np.max(f_story,axis = 1)
'''
l_list = np.linspace(1,maxite)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(l_list,times[-1,:],'g+',label = "Wall times ARS")
ax1.set_xlabel('Number of qubits ')
ax1.set_ylabel('Times[s]')
ax1.set_title('QSL time graph-Max')
ax1.set_yscale('log')
#ax1.set_yscale('log')
ax1.legend()
fig.show()
'''