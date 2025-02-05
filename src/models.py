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

    The state vector for pulsar n is:
        X^(n) = [delta_phi, delta_f, r, a, delta_epsilon_1, ..., delta_epsilon_M]^T

    with measurement equation
        delta_t^(n) = (1/f0)*delta_phi - r + (design row)*[delta_epsilon].

    """

    def __init__(self,df_psr): 
        """Initialize the StochasticGWBackgroundModel.

        Parameters
        ----------
        df_psr : DataFrame
            DataFrame containing pulsar information.


        
        Initialize per–pulsar parameters from a pandas DataFrame.
        
        The DataFrame is assumed to contain at least the following columns:
           - dim_M: integer, the number of design parameters for that pulsar.
           - gamma_p: the spin–noise damping rate.
           - sigma_p: the spin–noise white noise amplitude.
           - f0: the pulsar spin frequency.
           - sigma_t: the measurement noise standard deviation.
        
        These are stored as numpy arrays.
        """


    
        self.Npsr = len(df_psr)
        self.name = 'Stochastic GW background model'
        self.nx   = self.Npsr*(2+2) + df_psr['dim_M'].sum() # δφ, δf, r,a + M terms for each pulsar
        self.M    = df_psr.dim_M.values


    def set_global_parameters(self, params):
        """
        Set global parameters for the model.
        
        params is a dictionary that must include:
            'Delta_t': time step (float)
            'gamma_a': GW damping rate (float)
            'h2': mean-square GW strain (<h^2>) (float)
            'sigma_eps': white-noise amplitude for timing model parameters (float)
            'theta_mat': an (N x N) symmetric array of angular separations (radians)
                         between pulsars.
        """


        self.γp = params['γp'] #vector (Npsr,)
        self.σp = params['σp'] #vector, (Npsr,)
        self.γa = params['γa'] #scalar
        self.h2 = params['h2'] #scalar

        self.σeps = params['σeps'] #for now a scalar which we will tune to a large value. May need to be more adjustable and be e.g. length M per pulsar

        self.separation_angle_matrix = params['separation_angle_matrix']


        self.f0 = params['f0']
        self.σt = params['σt']
        
 
    # ---------------------- Static Helper Methods ---------------------------
    @staticmethod
    def _hellings_downs(θ):
        """
        Compute the Hellings–Downs function for an angle theta (in radians).
        For theta==0 the autocorrelation is defined to be 1.
        """
        # Compute x = (1 - cos(theta))/2. (If theta==0, x==0.)
        x = (1 - np.cos(θ)) / 2

        # For theta==0, define Gamma = 1.
        # Otherwise, Gamma(theta) = (3/2)*x*ln(x) - x/4 + 0.5.
        return np.where(np.isclose(θ, 0.0), 1.0,(3/2) * x * np.log(x) - x/4 + 0.5)



    @staticmethod
    def _compute_F_p(γp , dt):
        """
        Compute the 2x2 state–transition matrix for the (δφ, δf) block.
        """
        exp_term = np.exp(-γp * dt)
        return np.array([[1.0, (1 - exp_term) / γp],
                         [0.0, exp_term]])
    
    @staticmethod
    def _compute_F_a(γa, dt):
        """
        Compute the 2x2 state–transition matrix for the (r, a) block.
        """
        exp_term = np.exp(-γa * dt)
        return np.array([[1.0, (1 - exp_term) / γa],
                         [0.0, exp_term]])
    

   
    @staticmethod
    def _compute_Q_block(γ,  σ, dt):
        """
        Compute the 2x2 discretised noise covariance for a process with damping rate gamma 
        and white noise amplitude sigma.
        """
        exp_term = np.exp(-γ * dt)
        exp_2gamma_dt = np.exp(-2 * γ * dt)
        term1 = dt / γ**2 - 2*(1 - exp_term) / γ**3 + (1 - exp_2gamma_dt) / (2 * γ**3)
        term2 = (1 - exp_term) / γ**2 - (1 - exp_2gamma_dt) / (2 * γ**2)
        term3 = (1 - exp_2gamma_dt) / (2 * γ)
        return σ**2 * np.array([[term1, term2],
                                    [term2, term3]])



    def F_matrix(self,dt): # For a givve nrun of the filter, dt can vary between timesteps, but parameters do not. Set params accordingly

        """
        Build the overall state–transition matrix F (block–diagonal over pulsars).
        
        For pulsar n, the state block is:
          [ F_p^(n)    0         0 ]
          [    0      F_a       0 ]
          [    0       0    I_(M[n]) ]
        where F_p^(n) is 2x2 and F_a is 2x2 (common to all pulsars).

        """
        
        # Precompute F_a using global gamma_a and Delta_t.
        F_a = self._compute_F_a(self.γa, dt)
        # Build each pulsar block.
        F_blocks = [
            block_diag(
                self._compute_F_p(self.γp[i], dt),F_a,np.eye(self.M[i])
                )
            for i in range(self.Npsr)
        ]
        # Overall F is block–diagonal.
        F = block_diag(*F_blocks)
        return F


    def Q_matrix(self,dt):
        """
        Build the full process–noise covariance matrix Q.
        
        For each pulsar n the diagonal block is:
            Q^(n,n) = block_diag( Q_p^(n), Q_a^(n,n), Q_eps^(n) )
        where:
          - Q_p^(n) is a 2x2 spin–noise covariance (using gamma_p and sigma_p).
          - Q_a^(n,n) is a 2x2 GW noise covariance with
              sigma_a^(n,n) = sqrt((h2/6)*gamma_a)   (Gamma=1 for autocorrelation).
          - Q_eps^(n) is the timing–model covariance: sigma_eps^2 * Delta_t * I_(M[n]).
        
        Off–diagonal blocks (n != m) are zero except in the GW block, where for pulsars
        n and m one uses:
              sigma_a^(n,m)^2 = (h2/6)*gamma_a*Gamma(theta_nm),
        and places the corresponding 2x2 covariance in the GW blocks.
        """


        # First, determine the sizes for each pulsar block.
        block_sizes = [2 + 2 + int(self.M[i]) for i in range(self.Npsr)]  # spin (2) + GW (2) + design (M)
        print("block sizes = ", block_sizes)
        cum_sizes = np.concatenate(([0], np.cumsum(block_sizes)))
        print("cum_sizes = ", cum_sizes)
        total_size = cum_sizes[-1]
        Q = np.zeros((total_size, total_size)) #todo can just initialise this directly with the correct shape? 
        print(Q.shape)
        

        
        # A lambda to compute a GW block given a sigma value.
        Q_a_template = lambda sigma: self._compute_Q_block(self.γa, sigma, dt)
        
        # Diagonal blocks.
        print(self.M)
       
        for n in range(self.Npsr):
            print("Iterating over pulsar ", n), self.Npsr
            i0, i1 = cum_sizes[n], cum_sizes[n+1]
            print("i0, i1 = ", i0, i1)
            # Spin noise block for pulsar n.
            Q_p = self._compute_Q_block(self.γp[n], self.σp[n], dt)

            # GW block for pulsar n: autocovariance uses Gamma=1.
            σ_a_nn = np.sqrt((self.h2 / 6) * self.γa)
            Q_a_nn = Q_a_template(σ_a_nn)

            # Timing–model block: size M[n].
            M_dim = int(self.M[n])
            Q_eps = self.σeps**2 * dt * np.eye(M_dim)
            Q_block = block_diag(Q_p, Q_a_nn, Q_eps)
            Q[i0:i1, i0:i1] = Q_block
    
        # Off–diagonal blocks in the GW sector.
        for n in range(self.Npsr):
            for m in range(n+1, self.Npsr):

                # Use the provided theta_mat to compute the Hellings–Downs value.
                Γ  = self._hellings_downs(self.separation_angle_matrix[n, m]) #should pre compute this. 
                σ_a_nm = np.sqrt((self.h2 / 6) * self.γa * Γ)
                Q_a_nm = Q_a_template(σ_a_nm)

                # For pulsar n the GW block is located after the first 2 rows.
                i0_n = cum_sizes[n] + 2
                i1_n = i0_n + 2
                i0_m = cum_sizes[m] + 2
                i1_m = i0_m + 2
                Q[i0_n:i1_n, i0_m:i1_m] = Q_a_nm
                Q[i0_m:i1_m, i0_n:i1_n] = Q_a_nm.T
        return Q

    def H_matrix(self):
        """
        Build a list of measurement matrices H (one per pulsar).
        
        For pulsar n the measurement equation is:
           delta_t = (1/f0)*delta_phi - r + (design row)*[delta_epsilon],
        so that H^(n) is the row vector:
           [1/f0, 0, -1, 0, zeros(M[n])]
        (Here we assume a dummy design row of zeros.)
        """
        H_list = [
            np.concatenate((
                np.array([1.0 / self.f0[i], 0.0, -1.0, 0.0]),
                np.zeros(int(self.M[i]))
            )).reshape(1, -1)
            for i in range(self.Npsr)
        ]
        return H_list



    def R_matrix(self):
        """
        Build the measurement–noise covariance matrix R for the pulsars observed
        at a given epoch.
        
        observed_indices is an array–like list of indices into the pulsar list.
        For pulsar n, the measurement noise variance is (sigma_t[n])^2.
        """
        return self.σt**2 # For now just a scalar










