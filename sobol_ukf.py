import os
from ukf import ukf
import numpy as np
import scipy.stats as ss
from SALib.sample import saltelli
from SALib.analyze import sobol
import sys


""" fix the random generation seed """
np.random.seed(2022)


""" state and observation space dimension """
n = 3
m = 1


# labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
#           "FR","HR","IT","CY","LV","LT","LU","HU","MT",
#           "NL","AT","PL","PT","RO","SI","SK","FI","SE"]


""" input two letter label country """
label = 'ES' 
#label = str(sys.argv[1])


"""
load infection records
"""

covid_data = np.loadtxt('data/covid_'+label+'.txt')[0:154]
nobs=len(covid_data)
covid_time = np.linspace(0,nobs-1,nobs)
dt = covid_time[1] - covid_time[0]


"""
load life expectancy and population size
"""

import csv
csv_data = []
with open('data/configuracion.csv') as file_obj:
    reader = csv.reader(file_obj)
    for row in reader:
        csv_data.append(row)
        
import pandas as pd        
df = pd.DataFrame(csv_data)

this_country = df[df[0]==label][:]

N = float(this_country[3][this_country.index[0]])


"""
known parameters
"""
sigma = 1.0/5.0
gamma = 1.0/7.0
beta0 = 3.0**2*gamma
log_beta0 = np.log(beta0)


""" initialize UKF class """
ukf = UKF(n, m, nobs, dt, label)


def support(p):
        
    """ Parameter support. Necessary for twalk
    
    Parameters
    ----------
    None.

    Returns
    -------
    X     : TYPE = Boolean
            DESCRIPTION. True if the parameters are in the support
    """
    rt = True
    rt &= (-3.0 < p[0] < 2.0)
    rt &= (0.0 < p[1] < 10**3)
    rt &= (0.0 < p[2] < 10**3) 
    rt &= (0 < p[3] < N**2)
    rt &= (0 < p[4] < N**2)
    rt &= (0 < p[5] < N**2)
    rt &= (0 < p[6] < 10**8)
    rt &= (0 < p[7] < 1)            
    rt &= (0 < p[8] < 1)                
    return rt


def condCov(Pin):
        
    """ Adds nuggets of increasing magnitude to a covariance 
    matrix until the matrix is positive definite.
    """
    i = -8
    while np.any(np.linalg.eigvals(Pin + 10**i*np.eye(n)) <=0 ):
        i += 1 
        print(i)                           
    return Pin + 10**i*np.eye(n)


def is_pos_def(A):
    """ Check if square matrix is positive definite """    
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False 


def energy(q):
        
    """ compute the prior """
    log_prior = 0.0  
    # distribucion normal para log(beta)
    log_prior += ss.norm.logpdf(q[0],loc=log_beta0,scale=1)
    # # distribucion gamma para E(0)
    log_prior += ss.gamma.logpdf(q[1],1,1,scale=10**3)
    # # distribucion gamma para I(0)
    log_prior += ss.gamma.logpdf(q[2],1,1,scale=10**3)    
    # distribucion gamma para Q
    log_prior += ss.gamma.logpdf(q[3],1,1,scale=N)
    log_prior += ss.gamma.logpdf(q[4],1,1,scale=N)
    log_prior += ss.gamma.logpdf(q[5],1,1,scale=N)
    # distribucion gamma para R
    log_prior += ss.gamma.logpdf(q[6],1,1,scale=N)
    # distribucion beta para w
    log_prior += ss.beta.logpdf(q[7],1.1,1.1)      
    # distribucion beta para q    
    log_prior += ss.beta.logpdf(q[8],1.1,1.1)          


    """ reset ukf variables """
    ukf.resetUKF(q)           


    """ compute the likelihood """
    log_likelihood = 0.0
    for i in range(1,nobs):

        try:        
            if not is_pos_def(ukf.P_aposteriori):
                ukf.P_aposteriori = condCov(ukf.P_aposteriori)           

            ukf.timeUpdate(i-1, q)
            ukf.measurementUpdate(covid_data[i])    
            
            if ukf.y >= 0:
            
                term = ss.norm.logpdf(covid_data[i],loc=ukf.y,scale=ukf.P_y)[0][0]

            else:
                
                term = -10**10

        except: 
            term = -10**10
        log_likelihood += term
    #print (-log_likelihood-log_prior)
    return -log_likelihood-log_prior
    
    
 
if __name__=="__main__":
    
    
    """ Make sure output directory exists """
    directory = 'sobol_output/'+label
    if not os.path.exists(directory):
        os.makedirs(directory)

    #NECESITAS ESTA FUNCION
    def energy_evaluate(values):
        Y = np.zeros([values.shape[0]])    
        for i,p in enumerate(values):          
            Y[i]=energy(p)
        return Y    
    
    """
    Carry out Sobol sensitivity analysis in an hypecube of plus-minus twenty percent
    """

    
    problem = {
    'num_vars': 9,
    'names': ['beta', 'E0', 'I0', 'Sigma11', 'Sigma22', 'Sigma33', 'Gamma11','w','q'],
    'bounds': [    
        [-3.0, 0.0],
        [0, 10**3],
        [0, 10**3],    
        [0, N],
        [0, N],
        [0, N],    
        [0, N/10000],
        [0,1],
        [0,1]]        
    }
    
    param_values = saltelli.sample(problem, 2**5)
    Y = energy_evaluate(param_values)
    Si = sobol.analyze(problem, Y, print_to_console=False)
    np.savetxt(directory+'/S1.txt',Si['S1'])    
    np.savetxt(directory+'/S1_conf.txt',Si['S1_conf'])