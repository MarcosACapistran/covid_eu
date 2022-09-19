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


burnin = 2000
subsample = 100


""" input two letter label country """
labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
          "FR","HR","IT","CY","LV","LT","LU","HU","MT",
          "NL","AT","PL","PT","RO","SI","SK","FI","SE"]


#labels = ["BE","ES"] 


omegas = []
for i,label in enumerate(labels):
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
    n = int(len(chain)/7)
    samples = np.reshape(chain,(n,7))
    omegas.append(samples[burnin::subsample,5])
 
my_dict = dict(zip(labels,omegas))

fig, ax = plt.subplots(figsize=(12,4))
ax.boxplot(my_dict.values(),bootstrap=5000)
ax.set_xticklabels(my_dict.keys())
ax.set_title(r'$\omega$')
plt.savefig('figs/omegas.png')
