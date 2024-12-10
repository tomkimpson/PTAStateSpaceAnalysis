#This is the test file for src/gravitational_waves.py 
from src import gravitational_waves
import numpy as np 


"""Check the principal axes function works ok """
def test_principal_axes():
    
    N = 5 


    delta = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=N)
    alpha = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    psi = np.random.uniform(low=0.0,high=2*np.pi,size=N)

    for i in range(N):
            m,n = gravitational_waves.principal_axes(np.pi/2.0 - delta[i],alpha[i],psi[i]) # Get the principal axes of the GW

            #Check they are unit vectors
            np.testing.assert_almost_equal(np.linalg.norm(m), 1.0)
            np.testing.assert_almost_equal(np.linalg.norm(n), 1.0)



    #Check when source is in the plane that z-cpt is zero
    delta = 0.0 
    alpha = np.pi/2 
    psi = np.pi/6
    m,n = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi) # Get the principal axes of the GW
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




"""Check the polarisation tensors principle works  """
def test_polarisation_tensors():
    

    delta = np.pi/6
    alpha = np.pi/6
    psi = np.pi/6

    m,n                 = gravitational_waves.principal_axes(np.pi/2.0 - delta,alpha,psi) # Get the principal axes of the GW
    gw_direction        = np.cross(m,n)                               # The GW source direction. #todo: probably fast to have this not as a cross product - use cross product in unit test
    e_plus,e_cross      = gravitational_waves._polarisation_tensors(m.T,n.T)              # The p

   
    #See equation 2a,2d of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.104008
    e_cross_manual = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_cross_manual[i,j] = m[i]*n[j] +n[i]*m[j]

    e_plus_manual = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_plus_manual[i,j] = m[i]*m[j] -n[i]*n[j]


    #These are flattened vectors, lets check the items correspond
    np.testing.assert_almost_equal(e_plus, e_plus_manual.flatten())
    np.testing.assert_almost_equal(e_cross, e_cross_manual.flatten())
    




def test_h_amplitudes():

    #Everything should be zero if no GW
    h = 0 
    iota = np.pi/6
    hp,hx = gravitational_waves._h_amplitudes(h,iota)

    assert hp == 0.0 
    assert hx == 0.0


    #...and also if iota is extremal (zero or pi/2)  
    h = 1 
    iota = np.pi/2
    hp,hx = gravitational_waves._h_amplitudes(h,iota)

    assert hp == 1.0 
    np.testing.assert_almost_equal(hx, 0.0)


   
    h = 1 
    iota = 0.0
    hp,hx = gravitational_waves._h_amplitudes(h,iota)

    assert hp == 2.0 
    assert hx == -2.0


# """We get the scalar H_ij q^i q^j via some flattened vectors for speed 
# Lets check these match with the explicit expressions"""
# def test_hij_summation():

#     P   = system_parameters.SystemParameters(h=1e-10)  
#     PTA = pulsars.Pulsars(P)         
#     prefactor = gravitational_waves._prefactors(P.δ,P.α,P.ψ,PTA.q,PTA.q_products,P.h,P.ι, P.Ω)


#     #Manual, explicit
#     m,n                 = gravitational_waves.principal_axes(np.pi/2.0 - P.δ,P.α, P.ψ) # Get the principal axes of the GW
#     e_cross_manual = np.zeros((3,3))
#     for i in range(3):
#         for j in range(3):
#             e_cross_manual[i,j] = m[i]*n[j] +n[i]*m[j]

#     e_plus_manual = np.zeros((3,3))
#     for i in range(3):
#         for j in range(3):
#             e_plus_manual[i,j] = m[i]*m[j] -n[i]*n[j]

#     hp,hx               = gravitational_waves._h_amplitudes(P.h,P.ι)
#                            # plus and cross amplitudes. Shape (K,)


#     Hij = hp*e_plus_manual + hx*e_cross_manual #this is shape(3,3)
#     scalar_values = np.zeros((PTA.Npsr)) #for each pulsar, there is a different H_ij q^i q^j
#     for k in range(PTA.Npsr):
#         scalar_value = 0.0
#         for i in range(3):
#             for j in range(3):
#                 value = Hij[i,j]*PTA.q[k,i]*PTA.q[k,j]
#                 scalar_value += value

#         scalar_values[k] = scalar_value
#     manual_prefactor = -scalar_values/(2*P.Ω*(1+np.dot(PTA.q,np.cross(m,n))))
#     np.testing.assert_array_almost_equal(prefactor,manual_prefactor)


