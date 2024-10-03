import bilby
import sys
import numpy as np 
import sys
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):

 
        ll,xres,yres = self.model.likelihood(self.parameters)
        
        return ll
    
            
def BilbySampler(KalmanFilter,init_parameters,priors,label,outdir,nlive):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

 
    # #Run the sampler
    print("RUN THE SAMPLER")


    result = bilby.run_sampler(likelihood, priors, 
                              label = label,
                              outdir=outdir,
                              sampler ='dynesty',
			                  sample='rwalk_dynesty',
                              check_point_plot=False,
                              npoints=nlive,
                              dlogz=0.1, #1e-6?
                              npool=1,
			                  plot=False,resume=True)

    return result
