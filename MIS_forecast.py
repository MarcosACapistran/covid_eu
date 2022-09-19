# coding=utf-8
import os
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
from ukf_plot import ukf
import corner
import sys


""" fix the random generation seed """
np.random.seed(2022)


""" state and observation space dimension """
n = 3
m = 1


labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
          "FR","HR","IT","CY","LV","LT","LU","HU","MT",
          "NL","AT","PL","PT","RO","SI","SK","FI","SE"]


""" input two letter label country """
#label = 'BG' 
#label = str(sys.argv[1])


"""
load infection records
"""
for label in labels:
   fore = np.loadtxt('fore/'+label+'/fore.txt')[0:15]
   aux=np.mean(fore, axis=0)

   f = open("fore/fore_MSI.txt","a")
   value = ''.join(str(aux))
   f.write(' '+value+' ')
   f.close()
  


    
    
