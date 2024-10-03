




from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import LinearModel
from kalman_filter import KalmanFilter
from bilby_wrapper import BilbySampler
from priors import priors_dict,bilby_priors_dict
from bilby_wrapper import BilbyLikelihood

import numpy as np
import bilby
import sys

import logging 
#logging.basicConfig()
#logging.getLogger(name="KalmanGW").setLevel(logging.INFO)


arg_name = sys.argv[1]  #reference name
h        =  float(sys.argv[2]) #strain
measurement_model =  sys.argv[3] #whether to use the H0 or H1 model
seed = int(sys.argv[4]) #the seeding
nlive = int(sys.argv[5]) if len(sys.argv) >= 6 else 1000



from numba import jit, config


if __name__=="__main__":
    logger = logging.getLogger().setLevel(logging.INFO)
   
    import multiprocessing
    multiprocessing.set_start_method("fork")


    #Setup the system
    P   = SystemParameters(h=h,σp=None,σm=1e-11,use_psr_terms_in_data=True,measurement_model=measurement_model,seed=seed) # define the system parameters as a dict. Todo: make this a class
    PTA = Pulsars(P)                                       # setup the PTA
    data = SyntheticData(PTA,P)                            # generate some synthetic data

    #Define the model 
    model = LinearModel(P)

    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.f_measured,PTA)

    #Run the KF once with the correct parameters.
    #This allows JIT precompile
    optimal_parameters = priors_dict(PTA,P)
    model_likelihood, state_predictions,measurement_predictions = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")
    
    #Bilby
    init_parameters, priors = bilby_priors_dict(PTA,P)
   

    logging.info("Testing KF using parameters sampled from prior")
    params = priors.sample(1)
    model_likelihood, state_predictions,measurement_predictions = KF.likelihood(params)
    logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    # #Now run the Bilby sampler
    BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/",nlive=nlive)
    print("***Completed OK***")







