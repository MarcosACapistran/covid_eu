import numpy as np
import scipy
import scipy.linalg   
from scipy import integrate
import scipy.stats as ss

class ukf:

    def __init__(self, n, m, nobs, dt, label):

        """ dimensions """
        self.n = n
        self.m = m
        self.nobs = nobs       
        self.dt = dt 
        
        """ known parameters """        
        
        import csv
        csv_data = []
        with open('data/configuracion.csv') as file_obj:
            reader = csv.reader(file_obj)
            for row in reader:
                csv_data.append(row)
        
        import pandas as pd        
        df = pd.DataFrame(csv_data)

        this_country = df[df[0]==label][:]

        self.N = float(this_country[3][this_country.index[0]])        
        
        self.sigma = 1.0/5.0
        self.gamma = 1.0/14.0
        self.Pobs = 1.0 # probabilidad de observacion
       
        """ UKF parameters """
        self.kappa = 0.0
        self.alfa = 0.001
        self.beta = 2.0
        self.lambda_ = (self.n + self.kappa) * self.alfa * self.alfa - self.n
        self.gamma_ukf = np.sqrt(self.n + self.lambda_)
        self.W0m = self.lambda_ / (self.n + self.lambda_)
        self.W0c = self.lambda_ / (self.n + self.lambda_) + (1.0 - self.alfa * self.alfa + self.beta)
        self.W = 1.0 / (2.0 * (self.n + self.lambda_))
        
        
    """ seir propagator """        
    def state_seir(self, i, xp, p):
        
        def rhs(t, x, p):
            """
            epidemic model right hand side
            """ 
            fx1 = np.exp(p[0])*(self.N*self.omega_k-x[0]-x[1]-x[2])*x[1]/(self.N*self.omega_k) - self.sigma*x[0]
            fx2 = self.sigma*x[0] - self.gamma*x[1]
            fx3 = self.gamma*x[1]
            return [fx1,fx2,fx3]        

        def propagator(i, x, p):

            ti = i*self.dt
            teval = np.array(np.linspace(ti,ti+self.dt,11))
            dtfino=teval[1]-teval[0]
            tf=ti+self.dt+dtfino
            return integrate.solve_ivp(rhs, (ti,tf), x, method='RK45', 
                                       t_eval=teval, args=(p,))
                    
        output = propagator(i, xp, p)
        return output.y
 
    
    """ observation operator """
    def output_seir(self, xi):
        """
        evaluate likelihood expected value
        """
        fx = self.sigma*xi
        return self.Pobs*np.trapz(fx,dx=1.0/10.0)
        
   
    """ reset variables necessary for MCMC """ 
    def resetUKF(self, p):
        """ autoregressive models """        
        self.omega_k = np.copy(p[5])
        self.q_k = np.copy(p[6])        
        self.omegas = [self.omega_k]
        self.qs = [self.q_k]                

        """ all vectors used in the ukf process """
        self.x_apriori = np.zeros((self.n,), dtype=float)
        self.x_aposteriori = np.zeros((self.n,), dtype=float)
        self.x_P = np.zeros((self.n,), dtype=float)
        self.y_P = np.zeros((self.m,), dtype=float)
        self.y = np.zeros((self.m,), dtype=float)

        """ covariance matrices """
        self.P_apriori = np.zeros((self.n, self.n), dtype=float)
        self.P_aprioriP = np.zeros((self.n, self.n), dtype=float)
        self.P_aposteriori = np.zeros((self.n, self.n), dtype=float)

        """ square root product of a given covariances """
        self.sP_aposteriori = np.zeros((self.n, self.n), dtype=float)

        """ clear sigma points """
        self.y_sigma = np.zeros((self.m, (2 * self.n + 1)), dtype=float)
        self.x_sigma = np.zeros((self.n, (2 * self.n + 1)), dtype=float)

        """ sigma points after passing through the function f/h """
        self.x_sigma_f = np.zeros((self.n, (2 * self.n + 1)), dtype=float)
        self.x_sigma_f_E_quad = np.zeros(((2 * self.n + 1), 11), dtype=float)

        """ cross covariances """
        self.P_xy = np.zeros((self.n, self.m), dtype=float)
        self.P_xyP = np.zeros((self.n, self.m), dtype=float)

        self.P_y = np.zeros((self.m, self.m), dtype=float)
        self.P_y_P = np.zeros((self.m, self.m), dtype=float)
        self.K = np.zeros((self.n, self.m), dtype=float)

        self.Q = np.zeros((self.n, self.n), dtype=float)
        self.R = np.zeros((self.m, self.m), dtype=float)        

        """ initialize state and covariances """
        
        self.setCovariances(p)
        
        self.x_aposteriori = x0
        
        if self.x_aposteriori[2] == 0:
		""" give low importance to initial state """
            for i in range(0, self.n):
                self.P_aposteriori[i, i] = 10**1  
                #self.P_aposteriori = self.Q
        else:
            for i in range(0, self.n):
                self.P_aposteriori[i, i] = 10**-16  
                #self.P_aposteriori = self.Q

    def setCovariances(self, p):

        for i in range(self.n):
            self.Q[i, i] = p[1+i]    

        for i in range(self.m):
            self.R[i, i] = p[4]
            

    def sigma_points(self, vect_X, matrix_S):
        """ vect_X - state vector """
        """ sigma points are drawn from P """
        self.x_sigma[:, 0] = vect_X  # the first column

        for k in range(1, self.n+1):
            self.x_sigma[:, k] = vect_X + self.gamma_ukf * matrix_S[:, k - 1]
            self.x_sigma[:, self.n + k] = vect_X - self.gamma_ukf * matrix_S[:, k - 1]


    def y_UKF_calc(self):
        for j in range(2 * self.n + 1):
            """
            xi: for each sigmapoint, value of E in the 11 integration points
            """
            xi = self.x_sigma_f_E_quad[j, :]
            self.y_sigma[:, j] = self.output_seir(xi)

        """ y_UKF """
        self.y = self.W0m * self.y_sigma[:, 0]

        for k in range(1, 2 * self.n + 1):
            self.y = self.y + self.W * self.y_sigma[:, k]


    def state(self, w, theta):
        """ w - input vector data """
        for j in range(2 * self.n + 1):
            xp = self.x_sigma[:, j]
            aux = self.state_seir(w, xp, theta)
            self.x_sigma_f[:, j] = aux[:,-1]
            self.x_sigma_f_E_quad[j,:] = self.q_k*aux[0,:]
            

    def squareRoot(self, in_):
        out_ = scipy.linalg.cholesky(in_, lower=False)
        return out_


    def timeUpdate(self, w, theta):

        self.sP_aposteriori = self.squareRoot(self.P_aposteriori)

        self.sigma_points(self.x_aposteriori, self.sP_aposteriori)

        self.state(w, theta)

        """" apriori state """
        self.x_apriori= self.W0m * self.x_sigma_f[:, 0]
        for k in range(1, 2 * self.n + 1):
            self.x_apriori = self.x_apriori + self.W * self.x_sigma_f[:, k]


        """" apriori covariance matrix """
        self.P_apriori = np.zeros((self.n, self.n))

        for k in range(2 * self.n + 1):
            self.x_P = self.x_sigma_f[:, k]

            self.x_P = self.x_P - self.x_apriori
            self.P_aprioriP = np.dot(np.expand_dims(self.x_P, axis=1), np.transpose(np.expand_dims(self.x_P, axis=1)))

            if k == 0:
                self.P_aprioriP = np.dot(self.W0c, self.P_aprioriP)
            else:
                self.P_aprioriP = np.dot(self.W, self.P_aprioriP)
            self.P_apriori = self.P_apriori + self.P_aprioriP

        self.P_apriori = self.P_apriori + self.Q

        self.y_UKF_calc()


    def measurementUpdate(self, z):
        """" covariance matrix input/output """
        self.P_y = np.zeros((self.m, self.m))

        for k in range(2 * self.n + 1):
            self.y_P = self.y_sigma[:, k]

            self.y_P = self.y_P - self.y
            self.P_y_P = np.dot(np.expand_dims(self.y_P, axis=1), np.transpose(np.expand_dims(self.y_P, axis=1)))

            if k == 0:
                self.P_y_P = np.dot(self.W0c, self.P_y_P)
            else:
                self.P_y_P = np.dot(self.W, self.P_y_P)
            self.P_y = self.P_y + self.P_y_P

        self.P_y = self.P_y + self.R


        """ cross covariance matrix input/output """
        self.P_xy = np.zeros((self.n, self.m))

        for k in range(2 * self.n + 1):
            self.x_P = self.x_sigma_f[:, k]
            self.y_P = self.y_sigma[:, k]

            self.x_P = self.x_P - self.x_apriori
            self.y_P = self.y_P - self.y
            self.P_xyP = np.dot(np.expand_dims(self.x_P, axis=1), np.transpose(np.expand_dims(self.y_P, axis=1)))

            if k == 0:
                self.P_xyP = np.dot(self.W0c, self.P_xyP)
            else:
                self.P_xyP = np.dot(self.W, self.P_xyP)
            self.P_xy = self.P_xy + self.P_xyP

        """" kalman gain """
        self.K = np.dot(self.P_xy, np.linalg.inv(self.P_y))

        """ aposteriori state """
        self.y_P = z - self.y
        self.x_aposteriori = self.x_apriori + np.dot(self.K, self.y_P)

        """ cov aposteriori """
        self.P_aposteriori = self.P_apriori - np.dot(np.dot(self.K, self.P_y), np.transpose(self.K))

        """ update autoregressive models """
        self.z_k = np.log(self.omega_k/(1-self.omega_k))
        self.z_k = ss.norm.rvs(loc=self.z_k,scale=0.1)
        self.omega_k = np.exp(self.z_k)/(1+np.exp(self.z_k))

        self.p_k = np.log(self.q_k/(1-self.q_k))
        self.p_k = ss.norm.rvs(loc=self.p_k,scale=0.1)
        self.q_k = np.exp(self.p_k)/(1+np.exp(self.p_k))        

        self.omegas.append(self.omega_k)
        self.qs.append(self.q_k)                