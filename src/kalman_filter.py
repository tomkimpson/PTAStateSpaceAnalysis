


import numpy as np 
#from gravitational_waves import gw_model

from numba import jit,config


from model import F_function,R_function,Q_function #H function is defined via a class init

import sys 

from scipy.stats import multivariate_normal




"""
The log likelihood, designed for diagonal matrices where S is considered as a vector
"""
@jit(nopython=True)
def log_likelihood(S,innovation):
    x = innovation / S 
    N = len(x)
    
    slogdet = np.sum(np.log(S)) # Uses log rules and diagonality of covariance "matrix"
    value = -0.5*(slogdet+innovation @ x + N*np.log(2*np.pi))
    return value


"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
@jit(nopython=True)
def update(x, P, observation,R,Xfactor,ephemeris):


    H = 1.0 - Xfactor

    y_predicted = H*x - Xfactor*ephemeris
    y    = observation - y_predicted
    S    = H*P*H + R 
    K    = P*H/S 
    xnew = x + K*y


    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I_KH = 1.0 - K*H
    Pnew = I_KH * P * I_KH + K * R * K

    
    #And get the likelihood
    l = log_likelihood(S,y)
    
    return xnew, Pnew,l,y_predicted


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
@jit(nopython=True)
def predict(x,P,F,Q): 
    xp = F*x
    Pp = F*P*F + Q  

    return xp,Pp



class KalmanFilter:
    """
    A class to implement the Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

    """

    def __init__(self,Model, Observations,PTA):

        """
        Initialize the class. 
        """

        self.model = Model
        self.observations = Observations
        self.dt = PTA.dt
        self.q = PTA.q
        self.t = PTA.t
        self.q = PTA.q
        self.q_products = PTA.q_products

        self.Npsr = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]
        self.NF = PTA.NF
        self.H_function = Model.H_function
        self.x0 =  self.observations[0,:] 


    def likelihood(self,parameters):


        #Extract parameter values
        #To do - remove try/except
        try: #for the H1 model, these are dicts
            omega_gw = parameters["omega_gw"].item() 
            phi0_gw  = parameters["phi0_gw"].item()
            psi_gw   = parameters["psi_gw"].item()
            iota_gw  = parameters["iota_gw"].item()
            delta_gw = parameters["delta_gw"].item()
            alpha_gw = parameters["alpha_gw"].item()
            h        = parameters["h"].item()
        except: #for the null model, these are floats, not dict entries
            omega_gw = parameters["omega_gw"]
            phi0_gw  = parameters["phi0_gw"]
            psi_gw   = parameters["psi_gw"]
            iota_gw  = parameters["iota_gw"]
            delta_gw = parameters["delta_gw"]
            alpha_gw = parameters["alpha_gw"]
            h        = parameters["h"]
 
        #Noise parameters
        sigma_m = parameters["sigma_m"] #dont need an .item(), we always pass it as a float, don't infer it
        
        
        #Bilby takes a dict rather than a class
        #For us this is annoying - map some quantities to be vectors
        f,fdot,gamma,d,sigma_p = map_dicts_to_vector(parameters)
        f_EM = f + np.outer(self.t,fdot) #ephemeris correction



        #Precompute all the transition and control matrices as well as Q and R matrices.
        #F,Q,R are time-independent functions of the parameters
        #T is time dependent, but does not depend on states and so can be precomputed
        R = R_function(sigma_m)
        Q = Q_function(gamma,sigma_p,self.dt)
        F = F_function(gamma,self.dt)

        #Initialise x and P
        x = self.x0 # guess that the intrinsic frequencies is the same as the measured frequency
        P = np.ones(self.Npsr)* sigma_m * 1e3 #Guess that the uncertainty in the initial state is a few orders of magnitude greater than the measurement noise
  

        #Precompute the influence of the GW
        #Agan this does not depend on the states and so can be precomputed
        X_factor = self.H_function(delta_gw,
                                    alpha_gw,
                                    psi_gw,
                                    self.q,
                                    self.q_products,
                                    h,
                                    iota_gw,
                                    omega_gw,
                                    d,
                                    self.t,
                                    phi0_gw
                                )

        #Initialise the likelihood
        likelihood = 0.0
              
       
        x,P,l,ypred = update(x,P, self.observations[0,:],R,X_factor[0,:],f_EM[0,:])
        likelihood +=l


        #Place to store results
        x_results = np.zeros((self.Nsteps,self.Npsr))
        y_results = np.zeros_like(x_results)
        x_results[0,:] = x
        y_results[0,:] = (1.0 - X_factor[0,:])*x - X_factor[0,:]*f_EM[0,:] 
        

        for i in np.arange(1,self.Nsteps):

            obs = self.observations[i,:]
            x_predict, P_predict   = predict(x,P,F,Q)
            x,P,l,ypred = update(x_predict,P_predict, obs,R,X_factor[i,:],f_EM[i,:])
      
            likelihood +=l

            x_results[i,:] = x
            y_results[i,:] = (1.0 - X_factor[i,:])*x - X_factor[i,:]*f_EM[i,:] 
           
            
            
        return likelihood,x_results,y_results


"""
Useful function which maps repeated quantities in the dictionary to a
vector. Is there a more efficient way to do this? 
"""
def map_dicts_to_vector(parameters_dict):

    f = np.array([val.item() for key, val in parameters_dict.items() if "f0" in key])
    fdot = np.array([val.item() for key, val in parameters_dict.items() if "fdot" in key])
    gamma = np.array([val.item() for key, val in parameters_dict.items() if "gamma" in key])
    d = np.array([val.item() for key, val in parameters_dict.items() if "distance" in key])
    sigma_p = np.array([val.item() for key, val in parameters_dict.items() if "sigma_p" in key])

    return f,fdot,gamma,d,sigma_p



