

import numpy as np
from numba import jit,config

import logging 
from gravitational_waves import gw_psr_terms,gw_earth_terms,null_model
import sys
class LinearModel:

    """
    A linear model of the state evolution
    
    """

    def __init__(self,P):

        """
        Initialize the class. 
        """
        if P.measurement_model == "null":
            logging.info("You are using just the null measurement model")
            self.H_function = null_model 
        elif P.measurement_model == "earth":
            logging.info("You are using the Earth terms measurement model")
            self.H_function = gw_earth_terms
        elif P.measurement_model == "pulsar":
            self.H_function = gw_psr_terms
        else:
            sys.exit("Measurement model not recognized. Stopping.")


#These functions have to be outside the class to enable JIT compilation
#Bit ugly, but works from a performance standpoint
#To do: clean up
"""
The diagonal F matrix as a vector
"""
@jit(nopython=True)
def F_function(gamma,dt):
    return np.exp(-gamma*dt)

"""
The diagonal Q matrix as a vector
"""
@jit(nopython=True)
def Q_function(gamma,sigma_p,dt):
    value = sigma_p**2 * (1. - np.exp(-2.0*gamma* dt)) / (2.0 * gamma)
    return value 
    
"""
The R matrix as a scalar
"""
@jit(nopython=True)
def R_function(sigma_m):
    return sigma_m**2
    
