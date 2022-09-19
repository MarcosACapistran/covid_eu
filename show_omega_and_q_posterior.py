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


omegas = []
q = []
for i,label in enumerate(labels):
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
    n = int(len(chain)/7)
    samples = chain #np.reshape(chain,(n,7))
    omegas.append(samples[burnin::subsample,5])
    q.append(samples[burnin::subsample,6])
 
#my_dict_omegas = dict(zip(labels,omegas))

fig, ax = plt.subplots(2,1,figsize=(16,8),sharex='col')
ax[0].boxplot(omegas,bootstrap=5000)
#ax[0].set_xticklabels(labels)
ax[0].set_title(r'$\omega$')
ax[1].boxplot(q,bootstrap=5000)
#ax[1].set_xticklabels(labels)
ax[1].set_title(r'$q$')
plt.sca(ax[1])
plt.xticks(range(28),[" "]+labels)
plt.savefig('figs/omegas_and_qs.png')
