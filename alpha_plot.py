# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:00:01 2019

@author: 95juj
"""

import numpy as np
from matplotlib import pyplot as plt
plt.close('all')

num = 50
k_list = np.arange(0,num+1)
alpha_init_list = np.array([0.005,0.01])
beta_list = np.array([0.8,1,1.2])
leg_list = []

plt.figure()
for alpha_init in alpha_init_list:
    alpha = np.zeros(k_list.size)
    alpha[0] = alpha_init
    for beta in beta_list:
        for i,k in enumerate(k_list[1:]):
            alpha[i+1] = alpha[i] * (k)/(k+beta)
            
        plt.plot(k_list,alpha,'.-')
        leg_list.append("a_1 = {}, b = {}".format(alpha_init,beta))

plt.legend(leg_list)
plt.grid()
plt.xlabel('k')
plt.ylabel('a')
plt.show()

            