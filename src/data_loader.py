
from enterprise.pulsar import Pulsar as enterprise_pulsar
import numpy as np 



class LoadWidebandPulsarData():
    """A class to load and process pulsar data at a single frequency channel.

    This class provides methods to load pulsar data from parameter and timing files. 

    Attributes
    ----------
    toas : array-like
        Times of arrival of the pulsar signals.
    toaerrs : array-like
        Errors associated with the times of arrival.
    residuals : array-like
        Residuals of the pulsar timing model.
    fitpars : dict
        Fitted parameters of the pulsar timing model.
    toa_diffs : array-like
        Differences between consecutive times of arrival.
    toa_diff_errors : array-like
        Errors associated with the differences between consecutive times of arrival.
    M_matrix : array-like
        Design matrix for the pulsar timing model.
    name : str
        Name of the pulsar.

    Methods
    -------
    __init__(ds_psr):
        Initializes the LoadWidebandPulsarData object with pulsar data.
    read_par_tim(par_file, tim_file, **kwargs):
        Class method to load pulsar data from parameter and timing files using enterprise.

    """

    def __init__(self,ds_psr):
        """Initializes the LoadWidebandPulsarData object with pulsar data.

        This constructor initializes the object with various pulsar data attributes
        extracted from the provided pulsar data source object.

        Parameters
        ----------
        ds_psr : object
            An object containing pulsar data, typically an instance of a class that
            provides attributes such as toas, toaerrs, residuals, radio_frequencies,
            backend_flags, fitpars, and Mmat.

        """
        self.toas            = ds_psr.toas
        self.toaerrs         = ds_psr.toaerrs
        self.residuals       = ds_psr.residuals
        self.fitpars         = ds_psr.fitpars
        self.toa_diffs       = np.diff(self.toas)
        self.toa_diff_errors = np.sqrt(self.toaerrs[1:]**2 + self.toaerrs[:-1]**2)
        self.M_matrix        = ds_psr.Mmat
        self.name            = ds_psr.name

        self.RA = ds_psr._raj
        self.DEC = ds_psr._decj



    @classmethod #following structure in minnow, https://github.com/meyers-academic/minnow/blob/main/src/minnow/pulsar.py . Why not just pass parfile and timfile direct to __init__ ? 
    def read_par_tim(cls, par_file: str, tim_file: str, **kwargs) -> 'LoadWidebandPulsarData':
        """Loads the pulsar data from the specified file paths, uses enterprise to extract
        the relevant data, and stores it in the data attribute.

        Parameters
        ----------
        par_file : str
            Path to the parameter file.
        tim_file : str
            Path to the timing file.
        **kwargs : dict
            Additional keyword arguments to pass to the enterprise pulsar loader.

        Returns
        -------
        LoadWidebandPulsarData
            An instance of the LoadWidebandPulsarData class initialized with the loaded data.

        """
        try:
            pulsar_object = enterprise_pulsar(str(par_file), str(tim_file), **kwargs) #error handling. Probably overkill? 
            return cls(pulsar_object)
        except Exception as e:
            print(f"Error loading pulsar data: {e}")
            raise



   

