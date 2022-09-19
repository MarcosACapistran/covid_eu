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


for label in labels:
    data = np.loadtxt('data/covid_'+label+'.txt')[0:169]
    for i in np.where(data<0)[0]:
        data[i] = float(round(0.5*(data[i-1]+data[i+1])))
    np.savetxt('data/covid_'+label+'.txt',data)