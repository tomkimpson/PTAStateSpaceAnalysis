#This is the test file for src/gravitational_waves.py 
from src import gravitational_waves
import numpy as np 


"""Check the principal axes function works for a single BH-BH source """
def test_principal_axes_single_source():
    


    #Check that m/n vectors are unit vectors for M, individual, random sources
    M = 5 
    δ = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=M)
    α = np.random.uniform(low=0.0,high=2*np.pi,size=M)
    ψ = np.random.uniform(low=0.0,high=2*np.pi,size=M)

    for i in range(M):
            m,n = gravitational_waves._principal_axes(np.pi/2.0 - δ[i],α[i],ψ[i],M=1) # Get the principal axes of the GW

            #Check they are unit vectors
            np.testing.assert_almost_equal(np.linalg.norm(m), 1.0)
            np.testing.assert_almost_equal(np.linalg.norm(n), 1.0)



    #Check when source is in the plane that z-cpt is zero
    δ = 0.0 
    α = np.pi/2 
    ψ = np.pi/6
    m,n = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ,M=1) 
    gw_direction        = np.cross(m,n) 
    np.testing.assert_almost_equal(gw_direction[0,-1], 0.0)



    #Check when source is maximally high that the direction vector is [0,0,-1]
    δ = np.pi/2
    α = np.pi/2 
    ψ = np.pi/6
    m,n = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ,M=1) # Get the principal axes of the GW
    gw_direction        = np.cross(m,n) 
    assert np.all(gw_direction == np.array([0,0,-1]))





    #Check source location does not depend on psi
    δ = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=1)
    α = np.random.uniform(low=0.0,high=2*np.pi,size=1)
    ψ1= np.random.uniform(low=0.0,high=2*np.pi,size=1)
    m1,n1 = gravitational_waves._principal_axes(np.pi/2.0 - δ, α,ψ1,M=1)
    gw_direction1        = np.cross(m1,n1) 

  
    ψ2= np.random.uniform(low=0.0,high=2*np.pi,size=1)
    m2,n2 = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ2,M=1) 
    gw_direction2        = np.cross(m2,n2) 
    assert np.allclose(gw_direction1,gw_direction2) #same to within some float tolerance




"""Check the principal axes function works for multiples BH-BH sources"""
def test_principal_axes_multiple_source():

    #Check unit vectors
    M = 10
    δ = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=M)
    α = np.random.uniform(low=0.0,high=2*np.pi,size=M)
    ψ = np.random.uniform(low=0.0,high=2*np.pi,size=M)

    m,n = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ,M=M) # Get the principal axes of the GW
    for i in range(M):
        np.testing.assert_almost_equal(np.linalg.norm(m[i,:]), 1.0)
        np.testing.assert_almost_equal(np.linalg.norm(n[i,:]), 1.0)



    #Check source locations do not depend on psi
    ψ1= np.random.uniform(low=0.0,high=2*np.pi,size=M)
    m1,n1 = gravitational_waves._principal_axes(np.pi/2.0 - δ, α,ψ1,M=M)
    gw_direction1        = np.cross(m1,n1) 

  
    ψ2= np.random.uniform(low=0.0,high=2*np.pi,size=M)
    m2,n2 = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ2,M=M) 
    gw_direction2        = np.cross(m2,n2) 
    assert np.allclose(gw_direction1,gw_direction2) #same to within some float tolerance





"""Check the polarisation tensors function works for a single BH-BH binary"""
def test_polarisation_tensors_single_source():
    


    #Pick some arbitrary values
    δ = np.pi/6
    α = np.pi/6
    ψ = np.pi/6

    m,n                 = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ,M=1) # Get the principal axes of the GW                             # The GW source direction. #todo: probably fast to have this not as a cross product - use cross product in unit test
    e_plus,e_cross      = gravitational_waves._polarisation_tensors(m.T,n.T)              # The p

    #Calculate the tensor values explicity   
    #See equation 2a,2d of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.104008
    e_cross_manual = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_cross_manual[i,j] = m[0,i]*n[0,j] +n[0,i]*m[0,j]

    e_plus_manual = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            e_plus_manual[i,j] = m[0,i]*m[0,j] -n[0,i]*n[0,j]

    np.testing.assert_almost_equal(e_plus[:,:,0], e_plus_manual)
    np.testing.assert_almost_equal(e_cross[:,:,0], e_cross_manual)
    



"""Check the polarisation tensors function works, for a multiple BH-BH binaries"""
def test_polarisation_tensors_multiple_source():
    


    #Check unit vectors
    M = 10
    δ = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=M)
    α = np.random.uniform(low=0.0,high=2*np.pi,size=M)
    ψ = np.random.uniform(low=0.0,high=2*np.pi,size=M)

    m,n                 = gravitational_waves._principal_axes(np.pi/2.0 - δ,α,ψ,M=M) # Get the principal axes of the GW                             # The GW source direction. #todo: probably fast to have this not as a cross product - use cross product in unit test
    e_plus,e_cross      = gravitational_waves._polarisation_tensors(m.T,n.T)            


 

    for k in range(M):  #for each BH-BH binary, check the polarisation tensor explicitly

        #See equation 2a,2d of https://journals.aps.org/prd/abstract/10.1103/PhysRevD.81.104008
        e_cross_manual = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                e_cross_manual[i,j] = m[k,i]*n[k,j] +n[k,i]*m[k,j]

        e_plus_manual = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                e_plus_manual[i,j] = m[k,i]*m[k,j] -n[k,i]*n[k,j]

        np.testing.assert_almost_equal(e_plus[:,:,k], e_plus_manual)
        np.testing.assert_almost_equal(e_cross[:,:,k], e_cross_manual)



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




# import numpy as np

# # Sample data for H and q
# M = 2
# N = 2
# H = np.random.rand(3, 3, M)  # Example shape (3, 3, M)
# q = np.random.rand(3, N)     # Example shape (3, N)

# # Use np.einsum for the contraction
# result = np.einsum('ijm, in, jn -> mn', H, q, q)

# print('H is = ')
# print(H[:,:,0])
# print('-----')
# print(H[:,:,1])
# print('------_')
# print('q is =')
# print(q)
# print(result.shape)  # Should be (M, N)
# print(result)


# hflat = H[:,:,0].flatten()

# running_sum = 0
# for i in range(3):
#     for j in range(3):
#         running_sum += H[i,j,0]*q[i,0]*q[j,0]


# import numpy as np

# # Example data for H and q
# M = 5  # Depth of H
# N = 4  # Rows in q
# H = np.random.rand(3, 3, M)  # Shape (3, 3, M)
# q = np.random.rand(N, 3)     # Shape (N, 3)

# # Compute H_{ij} q^i q^j using np.einsum
# result = np.einsum('ijm,in,jn->mn', H, q.T, q.T)

# print(result.shape)  # Should be (M, N)
# print(result)




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


