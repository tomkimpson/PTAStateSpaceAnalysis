
"""Module used to generate fake, synthetic frequency timeseries for testing the pipeline"""


from numpy import sin, cos
import numpy as np 
import pandas as pd 



from pathlib import Path
import os 




class Pulsars:

    """
    A class which defines the pulsars which make up the PTA

    It takes the following arguments:

        `pulsar_file`: a path to a CSV which holds the ephemeris parameters of N pulsars  

        `γp`: A float which specifies the (inverse) mean reversion timescale of the Ornstein Uhlenbeck process. All pulsars take the same value

        `dt_weeks`: A float which specifies how frequently the pulsars are observed. All pulsars are observed at the same times.

        `Tobs_years`: A float which specifies the total observation span in years 

    ...and a placeholder **kwargs, which is not currently used. 

    """


    def __init__(self,pulsar_file,γp,dt_weeks,Tobs_years):



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



        self.γp        = np.ones_like(self.f) * γp   # for every pulsar let γ be the same 


        #Some useful reshaping for vectorised calculations later
        self.d          = self.d.reshape(self.Npsr,1)



        #and the cross products
        q_products = np.zeros((self.Npsr ,9))
        k = 0
        for n in range(self.Npsr ):
            k = 0
            for i in range(3):
                for j in range(3):
                    q_products[n,k] = self.q[n,i]*self.q[n,j]
                    k+=1
        q_products = q_products.T
        self.q_products=q_products




    
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
    return np.array([qx, qy, qz]).T





# """
# Given an angle α, return the correlation
# """
# def correlation_function(α):

#     with np.errstate(divide='ignore', invalid='ignore'): #ignore the errors that arise from taking np.log(0). These get replaced with 1s
#         bar = (1.0 - np.cos(α))/2
#         out = np.nan_to_num(1.0 + 3.0*bar * (np.log(bar) - (1.0/6.0)),nan=1.0) #replace nans with 1 for when α=0

#     return out


#GW signal 