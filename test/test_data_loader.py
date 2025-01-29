
from src import data_loader 
import os 
import glob 
import pytest

@pytest.mark.filterwarnings("ignore::Warning")
def test_load_pulsar():


    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    directory = os.path.join(script_dir, '../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset1/')

    # Normalize the path
    directory = os.path.normpath(directory)

    # Get all .par files in the directory
    par_files = glob.glob(os.path.join(directory, '*.par'))

    # Generate the corresponding .tim file paths
    file_pairs = [(par_file, par_file.replace('.par', '.tim')) for par_file in par_files]

    # Try to load the first N pulsars
    for par_file, tim_file in file_pairs[0:1]:
        psr = data_loader.LoadWidebandPulsarData.read_par_tim(par_file, tim_file, timing_package="pint", ephem="DE440", bipm_version="BIPM2019", clk="TT(BIPM2019)")
  