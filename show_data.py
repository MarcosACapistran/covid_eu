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

data = np.zeros((27,154))
for i,label in enumerate(labels):
    data[i,:] = np.loadtxt('data/covid_'+label+'.txt')[0:154]
    #data[i,:][data[i,:]<0] = 0
    #data[i,:] /= np.max(data[i,:])
    
import seaborn as sns
fig, ax = plt.subplots(1,1,figsize=(16,9))
xticks = np.arange(1,155)
sns.heatmap(data,cmap="coolwarm",xticklabels=10, yticklabels=labels)
ax.set_xlabel('Day',fontsize=18)
ax.set_ylabel('Country',fontsize=18)
plt.savefig('figs/data.png')
    