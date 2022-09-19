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


burnin = 5000
subsample = 100


""" input two letter label country """
labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
          "FR","HR","IT","CY","LV","LT","LU","HU","MT",
          "NL","AT","PL","PT","RO","SI","SK","FI","SE"]


#labels = ["BE","ES"] 


sigma_11 = []
sigma_22 = []
sigma_33 = []
for i,label in enumerate(labels):
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
    #n = int(len(chain)/7)
    samples = chain #np.reshape(chain,(n,7))
    sigma_11.append(np.log10(samples[burnin::subsample,1]))
    sigma_22.append(np.log10(samples[burnin::subsample,2]))
    sigma_33.append(np.log10(samples[burnin::subsample,3]))
 
my_dict_11 = dict(zip(labels,sigma_11))
my_dict_22 = dict(zip(labels,sigma_22))
my_dict_33 = dict(zip(labels,sigma_33))

fig, ax = plt.subplots(3,1,figsize=(16,8),sharex=True)
ax[0].boxplot(my_dict_11.values(),bootstrap=5000)
#ax[0].set_xticklabels(my_dict_11.keys())
ax[0].set_ylim(0,10)
ax[0].set_yticks((0,5,10))
ax[0].set_title(r'$\Sigma$')
#plt.savefig('figs/Sigma_11.png')

#fig, ax = plt.subplots(figsize=(12,4))
ax[1].boxplot(my_dict_22.values(),bootstrap=5000)
#ax[1].set_xticklabels(my_dict_22.keys())
#ax.set_title(r'$\Sigma_{22}$')
#plt.savefig('figs/Sigma_22.png')

#fig, ax = plt.subplots(figsize=(12,4))
ax[2].boxplot(my_dict_33.values(),bootstrap=5000)
#ax[2].set_xticklabels(my_dict_33.keys())
#ax[2].set_title(r'$\Sigma_{33}$')
plt.sca(ax[2])
plt.xticks(range(28),[" "]+labels)
plt.savefig('figs/Sigma.png')