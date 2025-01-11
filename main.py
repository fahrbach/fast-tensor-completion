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
"""
def main():
    colors = mpl.colormaps['tab10'].colors

    data_manager = TensorDataManager()
    #X = data_manager.generate_random_normal(shape=(50, 50, 50))
    X = data_manager.generate_random_cp(shape=(50, 50, 50), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=64)
    #X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))
    #X = data_manager.load_cardiac_mri()
    #X = data_manager.load_hyperspectral()

    output_path = data_manager.output_path
    print('X.shape:', X.shape)
    print('X.size:', X.size)
    print(data_manager.output_path)

    """
    sample_ratio = 0.10
    rank = 16
    epsilon = 0.1

    print('============ direct ==========')
    result = run_cp_completion(X, sample_ratio, rank, output_path)
    print('solve_times:', result.step_times_seconds)
    plt.plot(result.train_rres, label='direct', c=colors[0])
    plt.plot(result.test_rres, linestyle='dashed', c=colors[0])

    print('============ lifted ==========')
    result = run_lifted_cp_completion(X, sample_ratio, rank, output_path, epsilon=epsilon)
    print('solve_times:', result.step_times_seconds)
    plt.plot(result.train_rres, label='lifted', c=colors[1])
    plt.plot(result.test_rres, linestyle='dashed', c=colors[1])

    plt.grid()
    plt.legend()
    plt.yscale('log')
    #delta = 0.1
    #plt.ylim([0 - delta, 1 + delta])
    plt.show()


    # Running times
    result = run_cp_completion(X, sample_ratio, rank, output_path)
    plt.plot(result.step_times_seconds, label='direct', c=colors[0])

    result = run_lifted_cp_completion(X, sample_ratio, rank, output_path, epsilon=epsilon)
    plt.plot(result.step_times_seconds, label='lifted', c=colors[1])

    plt.grid()
    plt.legend()
    plt.show()

    return
    """

    # Full sweep. 
    sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    ranks = [1, 2, 4, 8, 16]
    for sample_ratio in sample_ratios:
        for rank in ranks:
            print('(sample_ratio, rank):', (sample_ratio, rank))
            #result = run_cp_completion(X, sample_ratio, rank, output_path)
            result = run_lifted_cp_completion(X, sample_ratio, rank, output_path, epsilon=0.001)


main()

