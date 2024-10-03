


import bilby

import numpy as np
import random
import logging 
logging.getLogger().setLevel(logging.INFO)


"""
Add  constant prior vector
"""
def add_to_bibly_priors_dict_constant(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = f
        i+= 1

    return init_parameters,priors



"""
Add  logarithmic prior vector
"""
def add_to_bibly_priors_dict_log(x,label,init_parameters,priors,lower,upper): #same lower/upper for every one
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.LogUniform(lower,upper, key)
        #logging.info(f"Sigma p true value is {key} {f}")
        
        i+= 1

    return init_parameters,priors


"""
Add uniform prior vector
"""
def add_to_bibly_priors_dict_uniform(x,label,init_parameters,priors,tol):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(f-np.abs(f*tol),f+ np.abs(f*tol), key)
        
        i+= 1

    return init_parameters,priors



"""
Helper function to use with priors_dict()
"""
def add_to_priors_dict(x,label,dict_A):



    i = 0
    for f in x:
        key = label+str(i)
        dict_A[key] = f
        i+= 1

    return dict_A


"""
Create a dict of parameters to be consumed by the Kalman likelihood function
"""
def priors_dict(pulsar_parameters,P):


   priors = dict({
               "omega_gw": P.Ω,
               "phi0_gw":P.Φ0,
               "psi_gw":P.ψ,
               "iota_gw":P.ι,
               "delta_gw":P.δ,
               "alpha_gw":P.α,
               "h": P.h})
   priors = add_to_priors_dict(pulsar_parameters.f,"f0",priors)
   priors = add_to_priors_dict(pulsar_parameters.fdot,"fdot",priors)
   priors = add_to_priors_dict(pulsar_parameters.d,"distance",priors)
   priors = add_to_priors_dict(pulsar_parameters.γ,"gamma",priors)
   priors = add_to_priors_dict(pulsar_parameters.σp,"sigma_p",priors)
   priors["sigma_m"]= pulsar_parameters.σm
  
   return priors




def set_prior_on_state_parameters(init_parameters,priors,f,fdot,σp,γ,d):



    init_parameters,priors = add_to_bibly_priors_dict_uniform(f,"f0",init_parameters,priors,tol=1e-10)      #uniform
    init_parameters,priors = add_to_bibly_priors_dict_uniform(fdot,"fdot",init_parameters,priors,tol=0.01) #uniform

    #If we set the true process noise to zero, then don't bother searching over this parameter
    if np.all(σp) == 0.0:
        logging.info('The true process noise is zero.')
        logging.info('Not setting a prior for σp') 
        init_parameters,priors = add_to_bibly_priors_dict_constant(σp,"sigma_p",init_parameters,priors)           #constant
    else:
        init_parameters,priors = add_to_bibly_priors_dict_log(σp,"sigma_p",init_parameters,priors,1e-21,1e-19) #log. 
    
    
    init_parameters,priors = add_to_bibly_priors_dict_constant(γ,"gamma",init_parameters,priors)           #constant
    init_parameters,priors = add_to_bibly_priors_dict_constant(d,"distance",init_parameters,priors) #distance not needed unless we are using the PSR model, which we are not using currently


    return init_parameters,priors 




def set_prior_on_measurement_parameters(init_parameters,priors,measurement_model,P):


    if measurement_model == "null": #set these as constants. Not used in the filter for the null model

        #We define the GW parameters for consistency but these are not actually used
        #- a bit hacky. Will need to clear this up, but doing it this way
        #lets us have a single H_function() type call

        logging.info('Using the null priors for the measurement model')


        init_parameters["omega_gw"] = None
        priors["omega_gw"] = 1.0


        init_parameters["phi0_gw"] = None
        priors["phi0_gw"] = 1.0  

        init_parameters["psi_gw"] = None
        priors["psi_gw"] = 1.0

        init_parameters["iota_gw"] = None
        priors["iota_gw"] =1.0 


        init_parameters["delta_gw"] = None
        priors["delta_gw"] =1.0


        init_parameters["alpha_gw"] = None
        priors["alpha_gw"] = 1.0


        init_parameters["h"] = None
        priors["h"] = 1.0

    else:

        logging.info('Using the GW priors for the measurement model')
        
        #Add all the GW quantities
        init_parameters["omega_gw"] = None
        priors["omega_gw"] = bilby.core.prior.LogUniform(1e-8, 1e-6, 'omega_gw')
   

        init_parameters["phi0_gw"] = None
        #priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'phi0_gw')
        priors["phi0_gw"] = bilby.core.prior.Uniform(0.0, 2*np.pi, 'phi0_gw')


        init_parameters["psi_gw"] = None
        priors["psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'psi_gw')
        #priors["psi_gw"] = bilby.core.prior.Uniform(np.pi/2, np.pi, 'psi_gw') #just for speed


        init_parameters["iota_gw"] = None
        priors["iota_gw"] = bilby.core.prior.Sine(0.0, np.pi, 'iota_gw')
        #priors["iota_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'iota_gw')


        init_parameters["delta_gw"] = None
        priors["delta_gw"] = bilby.core.prior.Cosine(-np.pi/2, np.pi/2, 'delta_gw')


        init_parameters["alpha_gw"] = None
        priors["alpha_gw"] = bilby.core.prior.Uniform(0.0, 2*np.pi, 'alpha_gw')
 

        init_parameters["h"] = None
        priors["h"] = bilby.core.prior.LogUniform(P.h/10.0, P.h*10.0, 'h')


    return init_parameters,priors 





# https://arxiv.org/pdf/2008.12320.pdf
def bilby_priors_dict(PTA,P):


    logging.info('Setting the bilby priors dict')


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()


    #Measurement priors
    init_parameters,priors = set_prior_on_measurement_parameters(init_parameters,priors,P.measurement_model,P) #h is provided to set the prior a few orders of magnitude either side.

    #State priors
    init_parameters,priors = set_prior_on_state_parameters(init_parameters,priors,PTA.f,PTA.fdot,PTA.σp,PTA.γ,PTA.d)

    #Noise priors
    init_parameters["sigma_m"] = None
    priors["sigma_m"] = 1e-11



    return init_parameters,priors
