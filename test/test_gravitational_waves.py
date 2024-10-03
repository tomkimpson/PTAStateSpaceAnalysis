#This is the test file for py_src/gravitational_waves.py 
import gravitational_waves as gravitational_waves #,system_parameters, pulsars 
import random 
import numpy as np 
from numpy import sin,cos
import numba as nb

def test_principal_axes():

    """Check the principal axes function"""
    N = 5
    thetas = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=N)
    phis = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    psis = np.random.uniform(low=0.0,high=np.pi,size=N)

    for i in range(N):
        m,n = gravitational_waves.principal_axes(thetas[i],phis[i],psis[i])

        #Check magnitudes
        np.testing.assert_almost_equal(np.linalg.norm(m), 1.0) #https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html
        np.testing.assert_almost_equal(np.linalg.norm(n), 1.0) 

        #Check directions
        direction_inferred = np.cross(m,n)
        direction_explicit = np.array([sin(thetas[i])*cos(phis[i]),sin(thetas[i])*sin(phis[i]),cos(thetas[i])])
        np.testing.assert_almost_equal(direction_explicit,-direction_inferred)



    #Check when source is in the plane that z-cpt is zero
    delta = 0.0 
    alpha = np.pi/2 
    psi = np.pi/6
    m,n = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi) 
    gw_direction        = np.cross(m,n) 
    np.testing.assert_almost_equal(gw_direction[-1], 0.0)


    #Check when source is maximally high that it is [0,0,-1]
    delta = np.pi/2
    alpha = np.pi/2 
    psi = np.pi/6
    m,n = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi) # Get the principal axes of the GW
    gw_direction        = np.cross(m,n) 
    assert np.all(gw_direction == np.array([0,0,-1]))


    #Check source location does not depend on psi
    delta = 0.0 
    alpha = np.pi/2 
    psi1 = np.pi/6
    m1,n1 = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi1) # Get the principal axes of the GW
    gw_direction1        = np.cross(m1,n1) 

    delta = 0.0 
    alpha = np.pi/2 
    psi2 = np.pi/4
    m2,n2 = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi2) # Get the principal axes of the GW
    gw_direction2        = np.cross(m2,n2) 
    assert np.allclose(gw_direction1,gw_direction2) #same to within some float tolerance

def test_polarisation_tensors():

    """Check the polarisation tensors are as expected """
    

    delta = np.pi/6
    alpha = np.pi/6
    psi = np.pi/6

    m,n                 = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi) 
    gw_direction        = np.cross(m,n)                              
    e_plus,e_cross      = gravitational_waves.polarisation_tensors(nb.typed.List(m),nb.typed.List(n))    

   
    #Calculate the products manually and compare it with the list comprehension method that we use in the main code
    e_cross_manual = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_cross_manual[i,j] = m[i]*n[j] +n[i]*m[j]

    e_plus_manual = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_plus_manual[i,j] = m[i]*m[j] -n[i]*n[j]


    np.testing.assert_almost_equal(e_plus, e_plus_manual)
    np.testing.assert_almost_equal(e_cross, e_cross_manual)
    
def test_h_amplitudes():

    #Everything should be zero if no GW
    h = 0 
    iota = np.pi/6
    hp,hx = gravitational_waves.h_amplitudes(h,iota)

    assert hp == 0.0 
    assert hx == 0.0


    #If we look at the binary edge on, we should only get the h plus mode
    h = 1 
    iota = np.pi/2
    hp,hx = gravitational_waves.h_amplitudes(h,iota)

    assert hp == 1.0 
    np.testing.assert_almost_equal(hx, 0.0)

    #If we look at the binary face on, we should see a mixture. see e.g. https://physics.stackexchange.com/questions/622692/specific-gravitational-wave-polarisation-direction-for-edge-on-binary-system
    h = 1 
    iota = 0.0
    hp,hx = gravitational_waves.h_amplitudes(h,iota)

    assert hp == 2.0 
    assert hx == -2.0

def test_gw_phases():


    Ω = 5e-7
    Φ0 = 0.20
    χ = np.array([2,4])  #2 pulsars
    t = np.arange(0,100) #100 times
    ΦEarth,ΦPSR = gravitational_waves.get_gw_phases(Ω,Φ0,χ,t)

    # ΦPSR has been reshaped to allow addition with ΦEarth
    # If we subtract off the chi contribution to ΦPSR we should be left with ΦEarth
    ΦEarth_manual = ΦPSR-χ #this is an array of shape (100,2), but each column has the same values - the shared Earth values
    np.testing.assert_almost_equal(-Ω*t+Φ0, ΦEarth_manual[:,0])
    np.testing.assert_almost_equal(-Ω*t+Φ0, ΦEarth_manual[:,1])


    #Check the shapes are what you expect
    assert ΦEarth.shape == (100,1)
    assert ΦPSR.shape == (100,2)


#def test_shared_terms():

    #LETS DO THIS ONCE WE HAVE P AND PTA DEFINED

    #gravitational_waves.shared_terms(Ω,Φ0,ψ,ι,δ,α,h,χ,q,t)


# def test_polarisation_tensor():





#     polarisation_tensors








# """Check the null model is all zeros as expected"""
# def test_null_model():
    
    
#     N = 5
#     for i in range(N):

#         H_factor = gravitational_waves.null_model(
#                                 np.random.uniform(),
#                                  np.random.uniform(),
#                                  np.random.uniform(),
#                                  np.random.uniform(size=20), #this is q
#                                  np.random.uniform(),
#                                  np.random.uniform(),
#                                  np.random.uniform(),
#                                  np.random.uniform(),
#                                  np.random.uniform(),
#                                  np.random.uniform(size=10), #this is t
#                                  np.random.uniform(),
#                                 )
#     assert np.all(H_factor==0)
