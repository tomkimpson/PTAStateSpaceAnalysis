
from src import data_loader 
import os 
import glob 
import pytest
import random

@pytest.mark.filterwarnings("ignore::Warning") # this does not seem to supress warnings. Not sure why not.
def test_load_pulsar():


    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    directory = os.path.join(script_dir, '../data/NANOGrav15year/wideband/')


    # Get all .par files in the directory
    par_files = sorted(glob.glob(os.path.join(directory, 'par/*.par')))
    tim_files = sorted(glob.glob(os.path.join(directory, 'tim/*.tim')))

    assert len(par_files) == len(tim_files) 




    # Combine par_files and tim_files into pairs
    file_pairs = list(zip(par_files, tim_files))
    # Select 5 random pairs
    random_pairs = random.sample(file_pairs, 5)

    # Check we can load the files with no errors
    for par_file, tim_file in random_pairs:

        try:
            psr = data_loader.LoadWidebandPulsarData.read_par_tim(par_file, tim_file, timing_package="pint", ephem="DE440", bipm_version="BIPM2019", clk="TT(BIPM2019)")
        except Exception as e:
            pytest.fail(f"Failed to load pulsar data from {par_file} and {tim_file} with error: {e}")