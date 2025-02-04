
from src import data_loader, models 
import os 
import glob 
import pytest
import random
import pandas as pd 
from functools import reduce 
import numpy as np 
def test_StochasticGWBackgroundModel():



    #Load some data to test on 
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the invariant directory path
    directory = os.path.join(script_dir, '../data/IPTA_MockDataChallenge/IPTA_Challenge1_open/Challenge_Data/Dataset2/')


    # Get all .par files in the directory
    par_files = sorted(glob.glob(directory + '*.par'))
    tim_files = sorted(glob.glob(directory + '*.tim'))

    assert len(par_files) == len(tim_files) 


    # Combine par_files and tim_files into pairs
    file_pairs = list(zip(par_files, tim_files))



    #Data loading block with some dataframe shenanigans. We should move the df stuff to src/

    dfs = []
    dfs_meta = []
    total_num_rows = 0 
    i = 0
    
    for par_file, tim_file in file_pairs[0:2]:
        psr = data_loader.LoadWidebandPulsarData.read_par_tim(par_file, tim_file)

        print(psr.M_matrix.shape)
        print(psr.fitpars)

        df = pd.DataFrame({'toas': psr.toas, f'residuals_{i}': psr.residuals})

        df_meta = pd.DataFrame({'name': [psr.name], f'dim_M': [psr.M_matrix.shape[-1]],f'RA': [psr.RA],f'DEC': [psr.DEC]})

        dfs.append(df)
        dfs_meta.append(df_meta)

        total_num_rows += len(df)
        i += 1
    


    merged_df = reduce(lambda left, right: pd.merge(left, right, on='toas', how='outer'), dfs)
    combined_df = pd.concat(dfs_meta, ignore_index=True)


    model = models.StochasticGWBackgroundModel(combined_df)

    θ = {'dt': 0.50,
     'γp': np.ones(len(combined_df)),
     'γa': 0.50,
     }
    F_array = model.F_matrix(θ)

    assert F_array.shape == (model.nx, model.nx)

    # def test_F_matrix(self):
    #     """Abstract method for the model state transition matrix F."""
    #     pass




