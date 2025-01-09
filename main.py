from cp_completion_solvers import *
from tensor_data_manager import TensorDataManager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

import time


"""
TODO:
    - Add verbose option for each solve
    - Support L2 regularization
"""
def main():
    colors = mpl.colormaps['tab10'].colors

    data_manager = TensorDataManager()
    X = data_manager.generate_random_normal(shape=(50, 50, 50))
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=64)

    #X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))
    #X = data_manager.load_cardiac_mri()
    #X = data_manager.load_hyperspectral()
    output_path = data_manager.output_path
    print('X.shape:', X.shape)
    print('X.size:', X.size)
    print(data_manager.output_path)

    sample_ratio = 0.01
    rank = 8
    result = run_lifted_cp_completion(X, sample_ratio, rank, output_path)
    print('============ direct ==========')
    result = run_cp_completion(X, sample_ratio, rank, output_path)
    print('train_losses:', result.train_losses)
    print('test_losses:', result.test_losses)
    return

    # Full sweep. 
    sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    ranks = [1, 2, 4, 8, 16]
    for sample_ratio in sample_ratios:
        for rank in ranks:
            print('(sample_ratio, rank):', (sample_ratio, rank))
            result = run_cp_completion(X, sample_ratio, rank, output_path)


main()

