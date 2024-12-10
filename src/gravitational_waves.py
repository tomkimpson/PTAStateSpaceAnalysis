"""Module which holds all functions which are related to the properties of the gravitational wave."""

from numpy import sin,cos 
import numpy as np 

#@njit(fastmath=True)
def principal_axes(θ,φ,ψ):
    """Calculate the two principal axes of the GW propagation.

    Args:
        θ (float): A scalar in radians, the polar angle of the GW source 
        φ (float): A scalar in radians, the azimuthal angle of the GW source 
        ψ (float): A scalar in radians, the polarisation angle of the GW source 

    Returns:
        m (ndarray):  A vector of length 3, corresponding to a principal axis of the GW
        n (ndarray):  A vector of length 3, corresponding to a principal axis of the GW

    """
    m1 = sin(φ)*cos(ψ) - sin(ψ)*cos(φ)*cos(θ)
    m2 = -(cos(φ)*cos(ψ) + sin(ψ)*sin(φ)*cos(θ))
    m3 = sin(ψ)*sin(θ)
    m = np.array([m1,m2,m3])

    n1 = -sin(φ)*sin(ψ) - cos(ψ)*cos(φ)*cos(θ)
    n2 = cos(φ)*sin(ψ) - cos(ψ)*sin(φ)*cos(θ)
    n3 = cos(ψ)*sin(θ)
    n = np.array([n1,n2,n3])

    return m,n


#@njit(fastmath=True)
def _polarisation_tensors(m, n):
    """Calculate the two polarisation tensors e_+, e_x.

    Args:
        m (ndarray): A vector of length 3, corresponding to a principal axis of the GW
        n (ndarray): A vector of length 3, corresponding to a principal axis of the GW

    Returns:
        e_plus  (ndarray): A 3x3 array corresponding to the + polarisation
        e_cross (ndarray): A 3x3 array corresponding to the x polarisation

    """
    # For e_+,e_x, Tensordot might be a bit faster, but list comprehension has JIT support
    # Note these are 1D arrays, rather than the usual 2D struture
    #todo: check these for speed up
    e_plus              = np.array([m[i]*m[j]-n[i]*n[j] for i in range(3) for j in range(3)]) 
    e_cross             = np.array([m[i]*n[j]+n[i]*m[j] for i in range(3) for j in range(3)])

    return e_plus,e_cross



def polarisation_tensors(m, n):
    """Alternative method to calculate the two polarisation tensors e_+, e_x.

    Args:
        m (ndarray): A vector of length 3, corresponding to a principal axis of the GW
        n (ndarray): A vector of length 3, corresponding to a principal axis of the GW

    Returns:
        e_plus  (ndarray): A 3x3 array corresponding to the + polarisation
        e_cross (ndarray): A 3x3 array corresponding to the x polarisation

    """
    x, y = m.shape

    #See e.g. https://stackoverflow.com/questions/77319805/vectorization-of-complicated-matrix-calculation-in-python
    ma = m.reshape(x, 1, y)
    mb = m.reshape(1, x, y)

    na = n.reshape(x, 1, y)
    nb = n.reshape(1, x, y)

    e_plus = ma*mb -na*nb
    e_cross = ma*nb +na*mb

    return e_plus,e_cross










#@njit(fastmath=True)
def _h_amplitudes(h,ι): 
    """Calculate the plus/cross amplitude components of the GW.

    Args:
        h (float): A scalar, the dimensionless GW amplitude
        ι (float): A scalar in radians, the inclination angle of the GW source

    Returns:
        h_plus  (float): The + component of the GW amplitude
        h_cross (float): The x component of the GW amplitude

    """
    return h*(1.0 + cos(ι)**2),h*(-2.0*cos(ι)) #hplus,hcross


# #@njit(fastmath=True)
# def _prefactors(delta,alpha,psi,q,q_products,h,iota,omega):

#     #Time -independent terms
#     m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi) # Get the principal axes of the GW
#     gw_direction        = np.cross(m,n)                               # The GW source direction. #todo: probably fast to have this not as a cross product - use cross product in unit test
#     e_plus,e_cross      = _polarisation_tensors(m.T,n.T)              # The polarization tensors. Shape (3,3,K)
#     hp,hx               = _h_amplitudes(h,iota)                       # plus and cross amplitudes. Shape (K,)
#     Hij                 = hp * e_plus + hx * e_cross                  # amplitude tensor. Shape (3,3,K)
#     H                   = np.dot(Hij,q_products)                      
#     dot_product         = 1.0 + q @ gw_direction
  
    
#     prefactor = -H/(2*omega*dot_product)
#     return prefactor #,dot_product





# """
# What is the GW modulation factor, including all pulsar terms?
# """
# #@njit(fastmath=True)
# def gw_psr_terms(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,χ):
#     prefactor = _prefactors(delta,alpha,psi,q,q_products,h,iota,omega)


#     omega_t = -omega*t
#     omega_t = omega_t.reshape(len(t),1) #Reshape to (T,1) to allow broadcasting. #todo, setup everything as 2d automatically


#     earth_term = np.sin(-omega_t + phi0)
#     pulsar_term = np.sin(-omega_t + phi0+χ)


#     return prefactor*(earth_term - pulsar_term)
   
  
# """
# What is the GW modulation factor, neglecting tje pulsar terms?
# """
# #@njit(fastmath=True)
# def gw_earth_terms(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,χ):
#     prefactor = _prefactors(delta,alpha,psi,q,q_products,h,iota,omega)

#     omega_t = -omega*t
#     omega_t = omega_t.reshape(len(t),1)

#     earth_term = np.sin(omega_t + phi0)

#     return prefactor*(earth_term)


# """
# The null model - i.e. no GW
# """
# #@njit(fastmath=True)
# def null_model(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,χ):
#     return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    




