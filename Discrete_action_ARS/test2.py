#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:27:12 2022

@author: lukas
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 12:00:21 2021

@author: lukas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:12:32 2021

@author: loklu
"""

import numpy as np
from ARS_discrete_a import ARSTrainer
import matplotlib.pyplot as plt
# Here is the code:
A = ARSTrainer()

#The QSL should be around 3.3 for spin chain and 2.4 for two-level.
T = 2

Noise = 0.01
N = 20
v = 0.3
alpha = 0.3
maxepochs = 20
max_ite = 50


f_story_B = np.zeros((maxepochs,max_ite))
times_B = np.zeros((maxepochs,max_ite))
M_list2 = np.zeros((20,max_ite))

T_list = []
N_list = []
for i in range(max_ite):
    #T = 2.8
    N += 5
    pulse = np.random.rand(N)
    o = 0
    while np.max(f_story_B[:,i]) < 0.95:
        if o == 0:
            o = 1
            T -= 0.7
        else:
            T += 0.1
        M_list2[:,i],__,f_story_B[:,i],times_B[:,i] = A.train(pulse,N,T,alpha,v,Noise = Noise,maxepochs=maxepochs)
        pulse = np.random.rand(N)
        print(i)
    T_list.append(T)
    N_list.append(N)
    
    

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(N_list,T_list,'g+',label = "Fidelity max Waveless")
ax1.set_xlabel('N interpolated over')
ax1.set_ylabel('T needed for QSL convergence')
ax1.set_title('QSL ARS discrete action 2-level system')
ax1.legend()



