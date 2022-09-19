# coding=utf-8
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')
from ukf_plot import ukf
import pandas as pd
import csv

""" fix the random generation seed """
np.random.seed(2022)


""" state and observation space dimension """
n = 3
m = 1


# labels = ["BE","BG","CZ","DK","DE","EE","IE","EL","ES",
#           "FR","HR","IT","CY","LV","LT","LU","HU","MT",
#           "NL","AT","PL","PT","RO","SI","SK","FI","SE"]
    
"""
known parameters
"""
sigma = 1.0/5.0
gamma = 1.0/14.0
beta0 = 3.0*gamma
log_beta0 = np.log(beta0)


burnin = 20000
subsample = 200
show_n_samples = 300
n_fore = 15

if __name__=="__main__":

    #fig,axs = plt.subplots(3,2,figsize=(16,8),gridspec_kw={'width_ratios': [3, 1]},sharex='col')
    fig,axs = plt.subplots(3,1,figsize=(14,8),sharex=True)
    
    """ input two letter label country """
    label = "ES"

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

    csv_data = []
    with open('data/configuracion.csv') as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            csv_data.append(row)
        

    df = pd.DataFrame(csv_data)
    this_country = df[df[0]==label][:]
    N = float(this_country[3][this_country.index[0]])
    
   
    ukf = UKF(n, m, nobs, dt, label)
    
    probs = np.loadtxt('mcmc_output/'+label+'/probs.dat') 
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
          
    # n = int(len(chain)/7)
    samples = chain #np.reshape(chain,(n,7)) 
   
    """ initialize UKF class """
    ukf = UKF(3, 1, nobs, dt, label) 
    solns_p = np.zeros((show_n_samples,7))
    solns_x = np.zeros((show_n_samples,3))     
    solns_y = np.zeros((show_n_samples,nobs))
    fore = np.zeros((show_n_samples,n_fore))    
    solns_map = np.zeros(nobs)    
    
    
    """ assimilate """
    for k in np.arange(show_n_samples): 
        solns_p[k,:] = samples[-k*subsample,:]
        
        """ reset ukf variables """
        E0 = int(ss.gamma.rvs(1,loc=1,scale=10)) 
        I0 = int(ss.gamma.rvs(1,loc=1,scale=10))         
        ukf.resetUKF(solns_p[k,:],np.array([E0,I0,0]))           
        for i in range(1,nobs):            

            ukf.timeUpdate(i-1, solns_p[k,:])
            ukf.measurementUpdate(covid_data[i])    
                            
            solns_y[k,i] = ukf.y
            
        solns_x[k,:] = ukf.x_aposteriori
        
        ukf.resetUKF(solns_p[k,:],solns_x[k,:])
        
        ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
        ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
        ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))

        ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
        ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
        ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        
        
        x = ukf.state_seir(nobs-1, solns_x[k,:], solns_p[k,:])
        fore[k,0] = solns_p[k,-1]*ukf.output_seir(x[0,:])
        #fore[k,0] += ss.norm.rvs(loc=0,scale=solns_p[k,4])        
        
        for i in range(1,n_fore):
            ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
            ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
            ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))

            ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
            ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
            ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        

            x = ukf.state_seir(nobs-1+i, x[:,-1], solns_p[k,:])
            fore[k,i] = solns_p[k,-1]*ukf.output_seir(x[0,:])             
            #fore[k,i] += ss.norm.rvs(loc=0,scale=solns_p[k,4])            

    
    """ store median, mean, standard deviation and quantiles """
    solns_median = np.median(solns_y,axis=0)
    solns_mean = np.mean(solns_y,axis=0)    
    solns_std = np.std(solns_y,axis=0)    
    solns_quant = np.quantile(solns_y,q=[0.05,0.95],axis=0)        
                        
    """ plot assimilation and forecast """
    axs[0].plot(covid_time_all[1:],covid_data_all[1:],'r.',label='Data')    
    axs[0].text(1, 1.1*covid_data.max(), label, fontsize=15)
    axs[0].errorbar(covid_time[1:],solns_median[1:],yerr=solns_quant[:,1:],fmt='.',ecolor='k',lw=0.5,label='Model')                
    #axs[0].plot(covid_time[1:],solns_median[1:],'b.',label='median')
    #axs[0,0].plot(covid_time[1:],solns_mean[1:],'g,',label='mean') 
    #axs[0].legend(loc=0) 

    fore = np.maximum(fore,0)           
    median_fore = np.median(fore,axis=0)
    mean_fore = np.mean(fore,axis=0)    
    yerr_fore = np.std(fore,axis=0)
    quant_fore = np.quantile(fore,q=[0.05,0.95],axis=0)
    #ax2.errorbar(nobs-1+l,median_fore,yerr=3*yerr,fmt='.',color='k',ecolor='k',lw=0.5) 
    #ax2.errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),median_fore,yerr=quant_fore,fmt='.',ecolor='k',lw=0.5,label='Forecast')                           
    axs[0].errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),mean_fore,yerr=quant_fore,fmt=' ',ecolor='g',lw=0.5,label='Forecast')                               
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.95, 1.15),
          ncol=1, fancybox=True, shadow=True)
    

    """ input two letter label country """
    label = "BE"

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

    csv_data = []
    with open('data/configuracion.csv') as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            csv_data.append(row)
        

    df = pd.DataFrame(csv_data)
    this_country = df[df[0]==label][:]
    N = float(this_country[3][this_country.index[0]])
    
   
    ukf = UKF(n, m, nobs, dt, label)
    
    probs = np.loadtxt('mcmc_output/'+label+'/probs.dat') 
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
          
    # n = int(len(chain)/7)
    samples = chain # np.reshape(chain,(n,7)) 
   
    """ initialize UKF class """
    ukf = UKF(3, 1, nobs, dt, label) 
    solns_p = np.zeros((show_n_samples,7))
    solns_x = np.zeros((show_n_samples,3))     
    solns_y = np.zeros((show_n_samples,nobs))
    fore = np.zeros((show_n_samples,n_fore))    
    solns_map = np.zeros(nobs)    
    
    
    """ assimilate """
    for k in np.arange(show_n_samples): 
        solns_p[k,:] = samples[-k*subsample,:]
        
        """ reset ukf variables """
        E0 = int(ss.gamma.rvs(1,loc=1,scale=10)) 
        I0 = int(ss.gamma.rvs(1,loc=1,scale=10))         
        ukf.resetUKF(solns_p[k,:],np.array([E0,I0,0]))           
        for i in range(1,nobs):            

            ukf.timeUpdate(i-1, solns_p[k,:])
            ukf.measurementUpdate(covid_data[i])    
                            
            solns_y[k,i] = ukf.y
            
        solns_x[k,:] = ukf.x_aposteriori
        
        ukf.resetUKF(solns_p[k,:],solns_x[k,:])
        
        ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
        ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
        ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))

        ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
        ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
        ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        
        
        x = ukf.state_seir(nobs-1, solns_x[k,:], solns_p[k,:])
        fore[k,0] = solns_p[k,-1]*ukf.output_seir(x[0,:])
        #fore[k,0] += ss.norm.rvs(loc=0,scale=solns_p[k,4])        
        
        for i in range(1,n_fore):
            ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
            ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
            ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))

            ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
            ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
            ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        

            x = ukf.state_seir(nobs-1+i, x[:,-1], solns_p[k,:])
            fore[k,i] = solns_p[k,-1]*ukf.output_seir(x[0,:])             
            #fore[k,i] += ss.norm.rvs(loc=0,scale=solns_p[k,4])
    
    """ store median, mean, standard deviation and quantiles """
    solns_median = np.median(solns_y,axis=0)
    solns_mean = np.mean(solns_y,axis=0)    
    solns_std = np.std(solns_y,axis=0)    
    solns_quant = np.quantile(solns_y,q=[0.05,0.95],axis=0)        
                        
    """ plot assimilation and forecast """
    axs[1].plot(covid_time_all[1:],covid_data_all[1:],'r.',label='Data')    
    axs[1].text(1, covid_data.max(), label, fontsize=15)
    axs[1].errorbar(covid_time[1:],solns_median[1:],yerr=solns_quant[:,1:],fmt='.',ecolor='k',lw=0.5,label='Model')                
    #axs[1].plot(covid_time[1:],solns_median[1:],'b.',label='median')
    #axs[2,0].plot(covid_time[1:],solns_mean[1:],'g.',label='mean') 

    fore = np.maximum(fore,0)           
    median_fore = np.median(fore,axis=0)
    mean_fore = np.mean(fore,axis=0)    
    yerr_fore = np.std(fore,axis=0)
    quant_fore = np.quantile(fore,q=[0.05,0.95],axis=0)
    #ax2.errorbar(nobs-1+l,median_fore,yerr=3*yerr,fmt='.',color='k',ecolor='k',lw=0.5) 
    #ax2.errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),median_fore,yerr=quant_fore,fmt='.',ecolor='k',lw=0.5,label='Forecast')                           
    axs[1].errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),mean_fore,yerr=quant_fore,fmt=' ',ecolor='g',lw=0.5,label='Forecast')                               
    #plt.legend(loc=0)
    axs[1].set_ylabel('Infections',fontsize=18)

    """ input two letter label country """
    label = "LU"

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

    csv_data = []
    with open('data/configuracion.csv') as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            csv_data.append(row)
        

    df = pd.DataFrame(csv_data)
    this_country = df[df[0]==label][:]
    N = float(this_country[3][this_country.index[0]])
    
   
    ukf = UKF(n, m, nobs, dt, label)
    
    probs = np.loadtxt('mcmc_output/'+label+'/probs.dat') 
    chain = np.loadtxt('mcmc_output/'+label+'/chain.dat')
          
    # n = int(len(chain)/7)
    samples = chain # np.reshape(chain,(n,7)) 
   
    """ initialize UKF class """
    ukf = UKF(3, 1, nobs, dt, label) 
    solns_p = np.zeros((show_n_samples,7))
    solns_x = np.zeros((show_n_samples,3))     
    solns_y = np.zeros((show_n_samples,nobs))
    fore = np.zeros((show_n_samples,n_fore))    
    solns_map = np.zeros(nobs)    
    
    
    """ assimilate """
    for k in np.arange(show_n_samples): 
        solns_p[k,:] = samples[-k*subsample,:]
        
        """ reset ukf variables """
        E0 = int(ss.gamma.rvs(1,loc=1,scale=10)) 
        I0 = int(ss.gamma.rvs(1,loc=1,scale=10))         
        ukf.resetUKF(solns_p[k,:],np.array([E0,I0,0]))           
        for i in range(1,nobs):            

            ukf.timeUpdate(i-1, solns_p[k,:])
            ukf.measurementUpdate(covid_data[i])    
                            
            solns_y[k,i] = ukf.y
            
        solns_x[k,:] = ukf.x_aposteriori            
            
        ukf.resetUKF(solns_p[k,:],solns_x[k,:])
        
        ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
        ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
        ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))

        ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
        ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
        ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        
        
        x = ukf.state_seir(nobs-1, solns_x[k,:], solns_p[k,:])
        fore[k,0] = solns_p[k,-1]*ukf.output_seir(x[0,:])
        #fore[k,0] += ss.norm.rvs(loc=0,scale=solns_p[k,4])        
        
        for i in range(1,n_fore):
            ukf.z_k = np.log(ukf.omega_k/(1-ukf.omega_k))
            ukf.z_k = ss.norm.rvs(loc=ukf.z_k,scale=0.1)
            ukf.omega_k = np.exp(ukf.z_k)/(1+np.exp(ukf.z_k))

            ukf.p_k = np.log(ukf.q_k/(1-ukf.q_k))
            ukf.p_k = ss.norm.rvs(loc=ukf.p_k,scale=0.1)
            ukf.q_k = np.exp(ukf.p_k)/(1+np.exp(ukf.p_k))        

            x = ukf.state_seir(nobs-1+i, x[:,-1], solns_p[k,:])
            fore[k,i] = solns_p[k,-1]*ukf.output_seir(x[0,:])             
            #fore[k,i] += ss.norm.rvs(loc=0,scale=solns_p[k,4])            

    
    """ store median, mean, standard deviation and quantiles """
    solns_median = np.median(solns_y,axis=0)
    solns_mean = np.mean(solns_y,axis=0)    
    solns_std = np.std(solns_y,axis=0)    
    solns_quant = np.quantile(solns_y,q=[0.05,0.95],axis=0)        
                        
    """ plot assimilation and forecast """
    axs[2].plot(covid_time_all[1:],covid_data_all[1:],'r.',label='Data')    
    axs[2].text(1, 0.85*covid_data.max(), label, fontsize=15)
    axs[2].errorbar(covid_time[1:],solns_median[1:],yerr=solns_quant[:,1:],fmt='.',ecolor='k',lw=0.5,label='Model')                
    #axs[2].plot(covid_time[1:],solns_median[1:],'b.',label='median')
    #axs[1,0].plot(covid_time[1:],solns_mean[1:],'g,',label='mean') 

    fore = np.maximum(fore,0)           
    median_fore = np.median(fore,axis=0)
    mean_fore = np.mean(fore,axis=0)    
    yerr_fore = np.std(fore,axis=0)
    quant_fore = np.quantile(fore,q=[0.05,0.95],axis=0)
    #ax2.errorbar(nobs-1+l,median_fore,yerr=3*yerr,fmt='.',color='k',ecolor='k',lw=0.5) 
    #ax2.errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),median_fore,yerr=quant_fore,fmt='.',ecolor='k',lw=0.5,label='Forecast')                           
    axs[2].errorbar(np.linspace(nobs,nobs+n_fore-1,n_fore),mean_fore,yerr=quant_fore,fmt=' ',ecolor='g',lw=0.5,label='Forecast')                               
    #plt.legend(loc=0)   
    axs[2].set_xlabel('Day',fontsize=18)
    
    #axs[2,1].set_xticklabels(bwq_labels_be)         
    plt.savefig("figs/ES_BE_LUnew.png")
    plt.close()