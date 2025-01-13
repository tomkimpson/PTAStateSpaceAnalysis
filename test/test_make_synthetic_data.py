
#This is the test file for src/make_synthetic_data.py 
from src import make_synthetic_data
import numpy as np 




"""Check that the values returned for the pulsar parameters are as expected"""
def test_reasonable_pulsar_values(request):
    root = request.config.rootdir


    PTA = make_synthetic_data.Pulsars(pulsar_file=f'{root}/data/NANOGrav_pulsars.csv', #todo, code this path properly
                                      γp = 1e-13,
                                      dt_weeks = 1,Tobs_years=10)                                     # setup the PTA

    #Check the range of f values
    assert np.all(PTA.f < 1000) #No pulsars should be above 1000 Hz
    assert ~np.all(PTA.f < 50) #or slower than 50Hz


    #Check the range of fdots 
    assert np.all(np.abs(PTA.fdot) < 1e-13) #spindowns should be small 


    #Check distances
    #All pulsars should be in range 0.1-7 kpc, after converting units
    c = 3e8
    pc = 3e16
    assert 0.1 <= np.all(PTA.d*c/pc) <= 7


    #Check all alphas, deltas in range 
    assert np.all(PTA.δ <= np.pi/2)
    assert np.all(PTA.δ >= -np.pi/2)

    assert np.all(PTA.α <= 2*np.pi)
    assert np.all(PTA.α >= 0)



"""Test that the q-direction unit vector is reasonable"""
def test_unit_vector():


    #Get 5 pulsars and randomly sample their positions
    N = 5 
    δ = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=N)
    α = np.random.uniform(low=0.0,high=2*np.pi,size=N)



    #Check the vector is unit magnitude
    m = make_synthetic_data._unit_vector(np.pi/2.0 - δ,α)
    for i in range(N):
        np.testing.assert_almost_equal(np.linalg.norm(m[i]), 1.0)


    #When delta is in the plane, the z component should = 0
    δ = np.array([0.0])
    α = np.random.uniform(low=0.0,high=2*np.pi,size=1)
    m = make_synthetic_data._unit_vector(np.pi/2.0 - δ,α)
    np.testing.assert_almost_equal(m[:,2], 0)

    #When delta is maximal z component should =1
    δ = np.array([np.pi/2])
    α = np.random.uniform(low=0.0,high=2*np.pi,size=1)
    m = make_synthetic_data._unit_vector(np.pi/2.0 - δ,α)
    np.testing.assert_almost_equal(m[:,2], 1)

