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


# labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
#           "FR","HR","IT","CY","LV","LT","LU","HU","MT",
#           "NL","AT","PL","PT","RO","SI","SK","FI","SE"]


""" input two letter label country """
#label = 'ES' 
label = str(sys.argv[1])


"""
load infection records
"""

covid_data_all = np.loadtxt('data/covid_'+label+'.txt')[0:169]
nobs_all=len(covid_data_all)
covid_time_all = np.linspace(0,nobs_all-1,nobs_all)

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
gamma = 1.0/14.0
beta0 = 3.0*gamma
log_beta0 = np.log(beta0)

if __name__=="__main__":
    
    
    """ Make sure output directory exists """
    directory = 'figs/'+label
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    """ Initialize the ukf filter class """
    ukf = UKF(n, m, nobs, dt, label)
    
    """ Set MCMC sampling parameters """ 
    burnin = 20000
    subsample = 200
    show_n_samples = 300
    n_fore = 15

    """ Load MCMC output """
    probs = np.loadtxt('mcmc_output/'+label+'/probs.dat') 
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
    
    samples = chain#np.reshape(chain,(n,7)) 
   
    """ Trace plot """
    fig,ax = plt.subplots()
    ax.plot(-probs[burnin::subsample])
    plt.title('Trace')
    plt.ylabel('Energy')
    plt.xlabel('Iteration')    
    plt.savefig(directory+'/trace.png') 
    plt.close()
        
    """ initialize UKF class """
    ukf = UKF(3, 1, nobs, dt, label) 
    solns_p = np.zeros((show_n_samples,7))
    solns_x = np.zeros((show_n_samples,3))     
    solns_y = np.zeros((show_n_samples,nobs)) 
    fore = np.zeros((show_n_samples,n_fore))       
    solns_map = np.zeros(nobs)    
    
    """ assimilate """
    k = 0
    n_success = 0
    while n_success < show_n_samples:
        try:
            solns_p[n_success,:] = samples[-k*subsample,:]
        
            """ reset ukf variables """
            E0 = int(ss.gamma.rvs(1,loc=1,scale=10)) 
            I0 = int(ss.gamma.rvs(1,loc=1,scale=10))  
                   
            ukf.resetUKF(solns_p[n_success,:],np.array([E0,I0,0]))           
            for i in range(1,nobs):            
                
                ukf.timeUpdate(i-1, solns_p[n_success,:])
                ukf.measurementUpdate(covid_data[i])    
                                
                solns_y[n_success,i] = ukf.y
            
            solns_x[n_success,:] = ukf.x_aposteriori
        
            ukf.resetUKF(solns_p[n_success,:],solns_x[n_success,:])
        
            ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
            ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
            ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))
            
            ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
            ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
            ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        
            
            x = ukf.state_seir(nobs-1, solns_x[n_success,:], solns_p[n_success,:])
            fore[n_success,0] = solns_p[k,-1]*ukf.output_seir(x[0,:])
            # uncomment next line to see forecasting with observation noise
            # fore[n_success,0] += ss.norm.rvs(loc=0,scale=solns_p[n_success,4])        
        
            for i in range(1,n_fore):
                ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
                ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
                ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))
                
                ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
                ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
                ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        
                
                x = ukf.state_seir(nobs-1+i, x[:,-1], solns_p[n_success,:])
                fore[n_success,i] = solns_p[n_success,-1]*ukf.output_seir(x[0,:]) 
                # uncomment next line to see forecasting with observation noise                
                # fore[n_success,i] += ss.norm.rvs(loc=0,scale=solns_p[n_success,4])
            n_success += 1
        except:
            pass
        k += 1
    
    """ store median, mean, standard deviation and quantiles """
    solns_median = np.median(solns_y,axis=0)
    solns_mean = np.mean(solns_y,axis=0)    
    solns_std = np.std(solns_y,axis=0)    
    solns_quant = np.quantile(solns_y,q=[0.05,0.95],axis=0)        
            
    """ plot assimilation and forecast """
    ax2.plot(covid_time_all[1:],covid_data_all[1:],'ro',label='Data')    
    ax2.text(1, covid_data.max(), label, fontsize=15)
    ax2.errorbar(covid_time[1:],solns_median[1:],yerr=solns_quant[:,1:],fmt='.',ecolor='k',lw=0.5,label='Model')                
      
    #fore = np.maximum(fore,0)  
    median_fore = np.median(fore,axis=0)
    mean_fore = np.mean(fore,axis=0)    
    yerr_fore = np.std(fore,axis=0)
    quant_fore = np.quantile(fore,q=[0.05,0.95],axis=0)

    ax2.errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),median_fore,yerr=quant_fore,fmt=' ',ecolor='g',lw=0.5,label='Forecast')                               
    plt.legend(loc=0) 

    plt.savefig(directory+'/data_vs_samples.png')
    plt.close()

    u = quant_fore[1,:]
    l = quant_fore[0,:]
    y = covid_data_all[nobs:nobs+n_fore]
    term1 = u-l
    term2 = (2/0.1)*(l-y)*(y<l)
    term3 = (2/0.1)*(y-u)*(y>u)
    fore_score = term1 + term2 + term3

    fore_score_mean=np.mean(fore_score, axis=0)
    fore_score=np.append(fore_score,fore_score_mean)
       
    fore_score_cover=100*np.sum((l<y)&(y<u))/n_fore
    fore_score=np.append(fore_score,fore_score_cover)

    
    """ Make sure output directory exists """
    directory_fore = 'fore/'+label
    if not os.path.exists(directory_fore):
        os.makedirs(directory_fore)    
    
    """ Store forecast """
    np.savetxt(directory_fore+'/fore.txt',fore_score)

    """ Make sure output directory exists """
    directory_asim = 'asim/'+label
    if not os.path.exists(directory_asim):
        os.makedirs(directory_asim)    
    
    """ Store assimilation """
    np.savetxt(directory_asim+'/asim.txt', (solns_median-covid_data)/np.max(covid_data))    
       
    """ Marginal posterior distributions of beta, q and omega"""         
    bwq_samples = np.copy(samples[burnin::subsample,[0,5,6]])
    bwq_samples[:,0] = np.exp(bwq_samples[:,0])
    mean = np.mean(bwq_samples,axis=0)
    std = np.std(bwq_samples,axis=0)
    plot_range = [(mean[x]-3*std[x],mean[x]+3*std[x]) for x in np.arange(3)]
    labels = [r"$\beta$", r"$\omega$", r"$q$"]   
    corner.corner(bwq_samples,
    labels=labels,
    range = plot_range,
    plot_datapoints=False,
    show_titles=True
    )
    plt.savefig(directory+'/corner_bwq.png')
    plt.close()

    """ Marginal posterior distribution of Sigma """
    mean = np.mean(samples[burnin::subsample,[1,2,3,4]],axis=0)
    std = np.std(samples[burnin::subsample,[1,2,3,4]],axis=0)
    plot_range = [(mean[x]-3*std[x],mean[x]+3*std[x]) for x in np.arange(4)]
    labels = [r"$\Sigma_{11}$", r"$\Sigma_{22}$", r"$\Sigma_{33}$",r"$\Gamma_{11}$"]                                
    corner.corner(samples[burnin::subsample,[1,2,3,4]],
    labels=labels,
    range = plot_range,
    plot_datapoints=False,
    show_titles=True
    )
    plt.savefig(directory+'/corner_SigmaGamma.png')    
    plt.close()

    """ R0 posterior distribution """
    plt.figure()
    R0 = np.sqrt(np.exp(samples[burnin::subsample,0])/(gamma*samples[burnin::subsample,5]))
    plt.axvline(R0.mean(), color='k', linestyle='dashed', linewidth=1,label='Media') 
    hist = plt.hist(R0,bins=100,density=True,label=r'$\mathcal{R}_0$')
    plt.legend(loc=0)
    plt.xlim(0.0,5.0)
    plt.savefig(directory+'/R0.png') 
    plt.close()
    
    """ Beta prior and posterior distribution """
    plt.figure()
    data = ss.norm.rvs(loc=log_beta0,scale=1.0,size=50000) 
    plt.hist(np.exp(data), 100, density=True,label='A priori '+r'$\pi_{B}(\beta)$')
    plt.hist(np.exp(samples[burnin::subsample,0]),bins=100,density=True,label='Posterior '+r'$\pi_{B|Z}(\beta|z)$',alpha=0.5)
    plt.legend(loc=0)
    plt.xlim(np.exp(-3.0),np.exp(1.0))
    plt.savefig(directory+'/beta_prior_vs_posterior.png')
    plt.close()

    
    
