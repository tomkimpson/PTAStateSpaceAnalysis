
from src import data_loader, models 
import os 
import glob 
import pytest
import random
import pandas as pd 
from functools import reduce 
import numpy as np 
from numpy import sin, cos


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

        #df_meta = pd.DataFrame({'name': [psr.name], f'dim_M': [psr.M_matrix.shape[-1]],f'RA': [psr.RA],f'DEC': [psr.DEC]})

        #just for testing with 2 pulsars, let M1 = 2 and M2 = 3
        df_meta = pd.DataFrame({'name': [psr.name], f'dim_M': 2+i, f'RA': [psr.RA],f'DEC': [psr.DEC]})



        dfs.append(df)
        dfs_meta.append(df_meta)

        total_num_rows += len(df)
        i += 1
    


    merged_df = reduce(lambda left, right: pd.merge(left, right, on='toas', how='outer'), dfs)
    combined_df = pd.concat(dfs_meta, ignore_index=True)
    
    print(combined_df)

    ra = combined_df['RA'].to_numpy()
    dec = combined_df['DEC'].to_numpy()



    """
    Given a latitude theta and a longitude phi, get the xyz unit vector which points in that direction 
    """
    def _unit_vector(theta,phi):
        qx = sin(theta) * cos(phi)
        qy = sin(theta) * sin(phi)
        qz = cos(theta)
        return np.array([qx, qy, qz]).T
    
    q = _unit_vector(np.pi/2.0 -dec, ra) # 3 rows, N columns
    Npsr = len(combined_df)


    #Get angle between all pulsars
    #Doing this explicitly for completeness - I am sure faster ways exist
    ζ = np.zeros((Npsr,Npsr))

    for i in range(Npsr):
        for j in range(Npsr):

            if i == j: #i.e. angle between same pulsars is zero
                ζ[i,j] = 0.0 
                
            else: 
                vector_1 = q[i,:]
                vector_2 = q[j,:]
                dot_product = np.dot(vector_1, vector_2)

                ζ[i,j] = np.arccos(dot_product)

    model = models.StochasticGWBackgroundModel(combined_df)


    #     # Set global parameters.
    params = {
        'γa': 0.001,      # s^-1
        'γp': np.ones(len(combined_df)),
        'σp': 1e-10*np.ones(len(combined_df)),
        'h2': 1e-12,
        'σeps': 1,
        'separation_angle_matrix': ζ,
        'f0': np.ones(len(combined_df)),
        'σt': 1e-1
    }


    model.set_global_parameters(params)
   
    dt = 0.50
    F = model.F_matrix(dt)
    Q = model.Q_matrix(dt)


    H = model.H_matrix()
    R = model.R_matrix()
    
    assert F.shape == (model.nx, model.nx)
    assert Q.shape == F.shape

 



