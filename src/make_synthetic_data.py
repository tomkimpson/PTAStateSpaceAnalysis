
"""Module used to generate fake, synthetic frequency timeseries for testing the pipeline."""
from numpy import sin, cos
import numpy as np 
import pandas as pd 
import bilby 
import logging 

#Set logging level for this module
logging.basicConfig(level=logging.INFO)

class BH_population:
    """ 
    Calculate the population of black holes which constitute the stochastic GW background.

    This involves randomly drawing 7 GW parameters (Ω,h,φ0,ψ,ι,α,δ) for M sources. 
    We use the bilby package to do do the random sampling; note that bilby does not currently let a user seed the sampling process.
    See e.g. https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/prior/analytical.py.

    """
    
    def __init__(self,Ω_power_law_index=None,Ω_min=None,Ω_max=None,M=None,parameters_dictionary=None):
        """Initialize the class."""
        #Assign arguments to class
        self.M = M
        logging.info(f'Generating a GW signal with M = {M}') 
     

        if (M == 1) and (parameters_dictionary is not None): #user specifies the parameters themselves by passing a dictionary
            
            #Manually extract from the dictionary and make them attributes of the class - easier to handle later
            self.Ω = parameters_dictionary['Ω']
            self.h = parameters_dictionary['h']
            self.φ0 = parameters_dictionary['φ0']
            self.ψ = parameters_dictionary['ψ']
            self.ι = parameters_dictionary['ι']
            self.δ = parameters_dictionary['δ']
            self.α = parameters_dictionary['α']

        else: # sample randomly

            #Assign arguments to class
            self.alpha = Ω_power_law_index
            self.Ω_min = Ω_min
            self.Ω_max = Ω_max

            #Define priors for GW parameters and sample
            priors  = self._gw_priors()
            samples = priors.sample(M)

            #Manually extract from the dictionary and make them attributes of the class - easier to handle later
            self.Ω = samples['Ω']
            self.h = samples['h']
            self.φ0 = samples['φ0']
            self.ψ = samples['ψ']
            self.ι = samples['ι']
            self.δ = samples['δ']
            self.α = samples['α']


  


    def _gw_priors(self):
        """Define the priors on the 7 GW parameters.

        Pass 3 arguments: Ω_power_law_index,Ω_min,Ω_max which define the power law prior on Ω.
        Note that h has a unit delta function prior - all sources have the same unit amplitude.
        """
        priors = bilby.core.prior.PriorDict()
        priors['Ω']  = bilby.core.prior.PowerLaw(alpha=self.alpha,minimum=self.Ω_min,maximum=self.Ω_max)
        priors['h']  = bilby.core.prior.DeltaFunction(1.0)
        priors['φ0'] = bilby.core.prior.Uniform(0.0, 2*np.pi)
        priors['ψ']  = bilby.core.prior.Uniform(0.0, np.pi)
        priors['ι']  = bilby.core.prior.Sine(0.0, np.pi)
        priors['δ']  = bilby.core.prior.Cosine(-np.pi/2, np.pi/2)
        priors['α']  = bilby.core.prior.Uniform(0.0, 2*np.pi)

        return priors







class Pulsars:
    """A class which defines the pulsars which make up the PTA.

    It takes the following arguments:

        `pulsar_file`: a path to a CSV which holds the ephemeris parameters of N pulsars  

        `γp`: A float which specifies the (inverse) mean reversion timescale of the Ornstein Uhlenbeck process. All pulsars take the same value of γp

        `dt_weeks`: A float which specifies how frequently the pulsars are observed. All pulsars are observed at the same times.

        `Tobs_years`: A float which specifies the total observation span in years 

    ...and a placeholder **kwargs, which is not currently used. 

    """

    def __init__(self,pulsar_file,γp,dt_weeks,Tobs_years):
        """Initialize the class."""
        #Define some universal constants
        pc = 3e16     # parsec in m
        c  = 3e8      # speed of light in m/s
        week = 7*24*3600 #a week in seconds


        #Read the CSV and Extract the parameters
        pulsars = pd.read_csv(pulsar_file)

        self.f         = pulsars["F0"].to_numpy()                    # Hz
        self.fdot      = pulsars["F1"] .to_numpy()                   # s^-2
        self.d         = pulsars["DIST"].to_numpy()*1e3*pc/c         # this is in units of s^-1
        self.δ         = pulsars["DECJD"].to_numpy()                 # radians
        self.α         = pulsars["RAJD"].to_numpy()                  # radians
        self.Npsr      = len(self.f) 
    
        #Pulsar positions as unit vectors
        self.q         = _unit_vector(np.pi/2.0 -self.δ, self.α) # 3 rows, N columns


        #For every pulsar let γ be the same 
        self.γp        = np.ones_like(self.f) * γp   


        #Some useful reshaping for vectorised calculations later
        self.d          = self.d.reshape(self.Npsr,1)



      
        # #It is useful in compute_a to have the products q1*q1, q1*q2 precomputed
        # #This lets us write h_ij q^i q&j as a single dot product
        # #There is probably a cleaner way to do this using np.einsum
        # #I am sure there is a better way to do this
        # self.q_products = np.zeros((self.Npsr,9))
        # k = 0
        # for n in range(self.Npsr):
        #     k = 0
        #     for i in range(3):
        #         for j in range(3):
        #             self.q_products[n,k] = self.q[n,i]*self.q[n,j]                    
        #             k+=1
        
        # self.q_products = self.q_products.T # lazy transpose here to enable correct shapes for dot products later




        # q_products = np.zeros((self.Npsr ,9))
        # k = 0
        # for n in range(self.Npsr ):
        #     k = 0
        #     for i in range(3):
        #         for j in range(3):
        #             q_products[n,k] = self.q[n,i]*self.q[n,j]
        #             k+=1
        # q_products = q_products.T
        # self.q_products=q_products




    
        # #Get angle between all pulsars
        # #Doing this explicitly for completeness - I am sure faster ways exist
        # self.ζ = np.zeros((self.Npsr,self.Npsr))

        # for i in range(self.Npsr):
        #     for j in range(self.Npsr):

        #         if i == j: #i.e. angle between same pulsars is zero
        #             self.ζ[i,j] = 0.0 
                    
        #         else: 
        #             vector_1 = self.q[i,:]
        #             vector_2 = self.q[j,:]
        #             dot_product = np.dot(vector_1, vector_2)

        #             self.ζ[i,j] = np.arccos(dot_product)

        # #Get the correlation between pulsar angles
        # self.pulsar_correlation = correlation_function(self.ζ)

     
        #Discrete timesteps
        self.dt = dt_weeks * week
        end_seconds  = Tobs_years* 365*24*3600 #from years to second
        self.t       = np.arange(0,end_seconds,self.dt)
      
  

        # # #Assign some other useful quantities to self
        # self.σm =  SystemParameters.σm
        # self.ephemeris = self.f + np.outer(self.t,self.fdot) 
     
        
        


"""
Given a latitude theta and a longitude phi, get the xyz unit vector which points in that direction 
"""
def _unit_vector(Θ,φ):
    qx = sin(Θ) * cos(φ)
    qy = sin(Θ) * sin(φ)
    qz = cos(Θ)
    return np.array([qx, qy, qz]).T #This has shape (N,3)





# """
# Given an angle α, return the correlation
# """
# def correlation_function(α):

#     with np.errstate(divide='ignore', invalid='ignore'): #ignore the errors that arise from taking np.log(0). These get replaced with 1s
#         bar = (1.0 - np.cos(α))/2
#         out = np.nan_to_num(1.0 + 3.0*bar * (np.log(bar) - (1.0/6.0)),nan=1.0) #replace nans with 1 for when α=0

#     return out


#GW signal 