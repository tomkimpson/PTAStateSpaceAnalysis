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

        self.nx = self.Npsr*(2+2) + df_psr['dim_M'].sum() # δφ, δf, r,a + M terms for each pulsar

        self.M = df_psr.dim_M.values


    def F_matrix(self,θ):
        """Compute the state transition matrix F for the model.

        Parameters
        ----------
        θ : dict
            Dictionary containing all free parameters of the model.

        Returns
        -------
        np.ndarray
            The state transition matrix F.

        """

        #Extract parameters
        dt = θ['dt'] # time between observations, scalar
        γp = θ['γp']  # frequency mean reversion timescale inverse, (N,)
        γa = θ['γa'] # a(t) mean reversion timescale inverse, scalar



        # Phase/Frequency block
        exp_γ = np.exp(-γp * dt)
        α     = (1 - exp_γ) / γp


        #Construct each 2x2 block for phase/frequency

        F_phase_blocks = [np.array([[1, a],
                                [0, b]]) for a, b in zip(α, exp_γ )]
        F_phase = block_diag(*F_phase_blocks)  # resulting shape: (2N, 2N)


        #Residuals/redshift block
        exp_γa = np.exp(-γa * dt)

        α_a = (1 -  exp_γa) / γa

        #Construct a 2x2 block for residuals/redshift. This is the same for every pulsar since γa is the same for each pulsar
        F_ra_single = np.array([[1,  α_a],
                                [0, exp_γa]])
        F_ra = np.kron(np.eye(self.Npsr), F_ra_single)  # resulting shape: (2N, 2N)

        # -- Mmatrix Block --
        F_offset_blocks = [np.eye(M_val) for M_val in self.M]
        F_offset = block_diag(*F_offset_blocks)



        F = block_diag(F_phase, F_ra, F_offset)

        return F
    
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

