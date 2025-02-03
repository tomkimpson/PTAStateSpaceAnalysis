from abc import ABC, abstractmethod
import numpy as np 


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




class ModelClass(ModelHyperClass): #concrete class

    def __init__(self,nx,ny):
        self.nx = nx #number of hidden states
        self.ny = ny #number of observations. Actually, this could change. Placeholder for now.
        pass


    @property
    def F_matrix(self):
        return np.eye(10)
    
    @property
    def Q_matrix(self):
        return np.eye(10)
    
    @property
    def H_matrix(self):
        return np.eye(10)


    @property
    def R_matrix(self):
        return np.eye(10)


c = ModelClass()  
print(c.R_matrix)