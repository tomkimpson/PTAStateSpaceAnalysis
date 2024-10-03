import system_parameters as system_parameters
import configparser

from configs.create_ini_file import create_ini
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

"""Check that we can create a config file, and that we have the same class variables from both initialisation methods"""
def test_config_creation():

    root = get_project_root()
    job_name    = 'unit_test'
    config_path = f'{root}/src/configs/{job_name}.ini'
    seed        = 1

    create_ini(config_path,job_name,seed=seed)

   
    P_config   = system_parameters.SystemParameters(config_path) #uses a config file
    P_default  = system_parameters.SystemParameters() #does not use a config file

    P_config_vars  = sorted(list(vars(P_default).keys()))
    P_default_vars = sorted(list(vars(P_config).keys()))

    assert P_config_vars == P_default_vars 



    #Check that the values in the config file match those in the SystemParameters class

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    #create a flattened dict across all sections
    flat_config_dict={}
    for each_section in config.sections():
        for (each_key, each_val) in config.items(each_section):
            flat_config_dict[each_key] = each_val


    for key,item in vars(P_config).items():
        try:
            float_var = eval(flat_config_dict[key]) #if the string can be converted to a float, do so
        except:
            float_var = flat_config_dict[key] #otherwise leave as a float
        assert item == float_var #check values in config match those in class



    