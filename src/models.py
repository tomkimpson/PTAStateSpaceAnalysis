"""Module for specifying models to be used with Kalman filter."""


from abc import ABC, abstractmethod
import numpy as np 
from scipy.linalg import block_diag

class ModelHyperClass(ABC): # abstract class
    """An abstract base class that defines the template for model classes.

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
        """Abstract method for the model state transition matrix F."""
        pass


    @property
    @abstractmethod
    def Q_matrix(self):
        """Abstract method for the model Q-matrix."""
        pass


    @property
    @abstractmethod
    def H_matrix(self):
        """Abstract method for the model H-matrix."""
        pass


    @property
    @abstractmethod
    def R_matrix(self):
        """Abstract method for the model R-matrix."""
        pass




class StochasticGWBackgroundModel(ModelHyperClass): #concrete class
    """A model class for the Stochastic Gravitational Wave Background.

    This class implements the required properties and methods for the
    Stochastic Gravitational Wave Background model.
    """

    def __init__(self,df_psr): 
        """Initialize the StochasticGWBackgroundModel.

        Parameters
        ----------
        df_psr : DataFrame
            DataFrame containing pulsar information.

        """
        self.Npsr = len(df_psr)
        self.name = 'Stochastic GW background model'

        self.nx = self.Npsr*(3+2) + df_psr['dim_M'].sum() # phi, f, fdot, a,r, + M terms for each pulsar

        self.M = df_psr.dim_M.values


    def F_matrix(self,θ):
        """Compute the state transition matrix F for the model.

        Parameters
        ----------
        θ : dict
            Dictionary containing model parameters, including 'dt' (time between observations).

        Returns
        -------
        np.ndarray
            The state transition matrix F.

        """
        dt = θ['dt'] #time between observations

        def _per_pulsar_F_matrix(M):
            """Return the F matrix for each pulsar.

            Parameters
            ----------
            M : int
                Dimension of the additional terms for each pulsar.

            Returns
            -------
            np.ndarray
                The state transition matrix F for a single pulsar.

            """
            Fφ = np.array([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
            F1 = np.eye(2+M)
            return block_diag(Fφ, F1)


        F_matrices = [_per_pulsar_F_matrix(M) for M in self.M]
        combined_matrix = block_diag(*F_matrices)

        assert combined_matrix.shape == (self.nx, self.nx)

        return combined_matrix
    
    @property
    def Q_matrix(self):
        """Compute the process noise covariance matrix Q.

        Returns
        -------
        np.ndarray
            The process noise covariance matrix Q.

        """
        return np.eye(10)
    
    @property
    def H_matrix(self):
        """Compute the measurement matrix H.

        Returns
        -------
        np.ndarray
            The measurement matrix H.

        """
        return np.eye(10)


    @property
    def R_matrix(self):
        """Compute the measurement noise covariance matrix Q.

        Returns
        -------
        np.ndarray
            The measurement noise covariance matrix Q.

        """
        return np.eye(10)

