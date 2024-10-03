import system_parameters,pulsars
from utils import pc,c
import pandas as pd 
import numpy as np 

    

"""Check that the pulsar parameters loaded from file are reasonable and within range"""
def test_pulsar_parameter_values():

    P  = system_parameters.SystemParameters()
    PTA = pulsars.Pulsars(P)


    #Check frequencies 
    assert np.max(PTA.f) < 1e3 # no pulsars faster than 1000 Hz
    assert np.min(PTA.f) > 10 # no pulsars slower than 10 Hz 


    #Check frequency spin down
    assert np.max(np.abs(PTA.fdot)) < 1e-11 
    assert np.min(np.abs(PTA.f)) > 1e-17



    #Check that if we convert our pulsar distances, which are in units of seconds, back to kpc we get reasonable values
    d_kpc = PTA.d*c/(1e3*pc)
    assert np.min(d_kpc) > 0.1 #no pulsars close away than 0.1 kpc
    assert np.max(d_kpc) < 10 #no pulsars further away than 10 kpc



    #Check all RAs are in the range
    assert np.all((PTA.α >= 0.0) & (PTA.α <= 2*np.pi))


    #Check all dects are in the range
    assert np.all((PTA.δ >= -np.pi/2) & (PTA.δ <= np.pi/2))



"""check that our q vectors satisfiy...what? #todo """
def check_q_vectors():
    pass