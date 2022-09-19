#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 20:17:50 2022

@author: marcos
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')


""" fix the random generation seed """
np.random.seed(2022)


burnin = 20000
subsample = 200


""" input two letter label country """
labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
          "FR","HR","IT","CY","LV","LT","LU","HU","MT",
          "NL","AT","PL","PT","RO","SI","SK","FI","SE"]


#labels = ["BE","ES"] 


betas = []
for i,label in enumerate(labels):
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
    n = int(len(chain)/7)
    samples = chain #np.reshape(chain,(n,7))
    betas.append(np.exp(samples[burnin::subsample,0]))
 
my_dict = dict(zip(labels,betas))

# fig, ax = plt.subplots(figsize=(12,4))
# ax.boxplot(my_dict.values(),bootstrap=5000)
# ax.set_xticklabels(my_dict.keys())
# ax.set_title(r'$\beta$')
# plt.savefig('figs/betas.png')


import seaborn as sns
fig, ax = plt.subplots(figsize=(20,6))
sns.boxplot(
    data=betas,
    color='red')
ax.set_xticklabels(my_dict.keys())
ax.set_xlabel('Country',fontsize=18)
ax.set_ylabel(r'$\beta$',fontsize=18)
ax.ylim=(0,4)
plt.savefig('figs/betas.png')
