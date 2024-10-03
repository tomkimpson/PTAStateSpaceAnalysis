

import sdeint
import numpy as np 

from gravitational_waves import gw_earth_terms,gw_psr_terms
import logging

class SyntheticData:
    
    

    def __init__(self,pulsars,P):


        #Load some PTA related quantities
        t = pulsars.t
        Npsr = pulsars.Npsr 

        #Pulsar parameters
        γ= pulsars.γ  
        σp= pulsars.σp  

        #Random seeding
        generator = np.random.default_rng(P.seed)

        
        #Turn σp and γ into diagonal matrices that can be accepted by vectorized sdeint
        σp = np.diag(σp)
        γ = np.diag(γ)

        #Integrate the state equation
        def f(x,t):
            return -γ.dot(x)
        def g(x,t):
            return σp

        self.intrinsic_frequency= sdeint.itoint(f,g,pulsars.fprime, t,generator=generator)

        #Now calculate the modulation factor due to the GW
        if P.use_psr_terms_in_data:
            GW_function = gw_psr_terms
            logging.info("You are including the PSR terms in your synthetic data generation")
        else:
            GW_function = gw_earth_terms
            logging.info("You are using just the Earth terms in your synthetic data generation")

        
        X_factor = GW_function(
                                        P.δ,
                                        P.α,
                                        P.ψ,
                                        pulsars.q,
                                        pulsars.q_products,
                                        P.h,
                                        P.ι,
                                        P.Ω,
                                        pulsars.d,
                                        pulsars.t,
                                        P.Φ0
                                        )
            
        #The measured frequency, no noise
        self.f_measured_clean= (1.0-X_factor)*self.intrinsic_frequency - X_factor*pulsars.ephemeris
        self.f_measured_clean_no_hetero= (1.0-X_factor)*self.intrinsic_frequency #- X_factor*pulsars.ephemeris

        measurement_noise = generator.normal(0, pulsars.σm,self.f_measured_clean.shape) # Measurement noise. Seeded
        self.f_measured = self.f_measured_clean + measurement_noise
        self.f_measured_no_hetero = self.f_measured_clean_no_hetero + measurement_noise
