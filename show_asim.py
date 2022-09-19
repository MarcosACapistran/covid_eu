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


""" input two letter label country """
labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
          "FR","HR","IT","CY","LV","LT","LU","HU","MT",
          "NL","AT","PL","PT","RO","SI","SK","FI","SE"]

#labels = ["BE","ES"] 

forecasts = np.zeros((27,154))
for i,label in enumerate(labels):
    forecasts[i,:] = np.loadtxt('asim/'+label+'/asim.txt') 
    
import seaborn as sns
fig, ax = plt.subplots(1,1,figsize=(16,9))
ax = sns.heatmap(-forecasts, xticklabels=10, yticklabels=labels, cmap="RdYlBu",vmin=-1, vmax=1)
ax.set_xlabel('Day',fontsize=18)
ax.set_ylabel('Country',fontsize=18)
plt.savefig('figs/assimilation_eval_scaled.png')