from abc import ABC, abstractmethod
import numpy as np 
from scipy.linalg import block_diag

class ModelHyperClass(ABC): # abstract class
    """
    An abstract base class that defines the template for model classes.

    This hyperclass specifies that any subclass must implement the following
    properties: F_matrix, Q_matrix, H_matrix, and R_matrix. These properties
    must be provided by any concrete implementation.

    Properties:
    -----------
    F_matrix : np.ndarray
        The state transition matrix. Subclasses must implement this property.
    
    Q_matrix : np.ndarray
        The process noise covariance matrix. Subclasses must implement this property.
    
    H_matrix : np.ndarray
        The observation matrix. Subclasses must implement this property.
    
    R_matrix : np.ndarray
        The observation noise covariance matrix. Subclasses must implement this property.
    """

    @property
    @abstractmethod
    def F_matrix(self):
        pass


    @property
    @abstractmethod
    def Q_matrix(self):
        pass


    @property
    @abstractmethod
    def H_matrix(self):
        pass


    @property
    @abstractmethod
    def R_matrix(self):
        pass




class StochasticGWBackgroundModel(ModelHyperClass): #concrete class

    def __init__(self,df_psr): #takes a dataframe of "meta" pulsar data i,e. a list of the pulsars used with other high level attrirbutes, rather than the raw TOAs/ residuals

        self.Npsr = len(df_psr)
        self.name = 'Stochastic GW background model'

        nx = self.Npsr*(3+2) + df_psr['dim_M'].sum() # phi, f, fdot, a,r, + M terms for each pulsar

        self.M = df_psr.dim_M.values


    def F_matrix(self,θ):

        dt = θ['dt'] #time between observations

        def _per_pulsar_F_matrix(M):
            """Return the F matrix for each pulsar."""
            Fφ = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
            F1 = np.eye(2+M)
            return block_diag(Fφ, F1)


        F_matrices = [_per_pulsar_F_matrix(M) for M in self.M]
        combined_matrix = block_diag(*F_matrices)

        return combined_matrix
    
    @property
    def Q_matrix(self):
        return np.eye(10)
    
    @property
    def H_matrix(self):
        return np.eye(10)


    @property
    def R_matrix(self):
        return np.eye(10)

