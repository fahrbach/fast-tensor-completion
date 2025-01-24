from cp_completion_solvers import *
from tensor_data_manager import TensorDataManager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl


def run_cp_completion_sweep(X, output_path):
    #seeds = [0, 1, 2, 3, 4, 5]
    seeds = [0]
    sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    #sample_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    ranks = [1, 2, 4, 8, 16]
    for seed in seeds:
        for sample_ratio in sample_ratios:
            for rank in ranks:
                print('Running... (seed, sample_ratio, rank):', (seed, sample_ratio, rank))
                result = run_cp_completion(X, sample_ratio, rank, output_path, seed=seed)
    return
    
    # Second plot
    sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ranks = [16]
    for seed in seeds:
        for sample_ratio in sample_ratios:
            for rank in ranks:
                print('Running... (seed, sample_ratio, rank):', (seed, sample_ratio, rank))
                result = run_cp_completion(X, sample_ratio, rank, output_path, seed=seed)


def run_lifted_cp_completion_sweep(X, output_path):
    #seeds = [0, 1, 2, 3, 4, 5]
    seeds = [0]

    #sample_ratios = [0.10]
    #sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    #sample_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    sample_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    epsilons = [0.1, 0.01, 0.001, 0.0001]
    rank = 16
    for seed in seeds:
        for sample_ratio in sample_ratios:
            for epsilon in epsilons:
                print('Running... (seed, sample_ratio, epsilon):', (seed, sample_ratio, epsilon))
                result = run_lifted_cp_completion(X, sample_ratio, rank, output_path, seed=seed, epsilon=epsilon)


def main():
    colors = mpl.colormaps['tab10'].colors

    data_manager = TensorDataManager()
    X = data_manager.generate_random_normal(shape=(100, 100, 100))
    output_path = data_manager.output_path
    print(data_manager.output_path)
    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)

    data_manager = TensorDataManager()
    X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    output_path = data_manager.output_path
    print(data_manager.output_path)
    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)

    data_manager = TensorDataManager()
    X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))
    output_path = data_manager.output_path
    print(data_manager.output_path)
    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)

    data_manager = TensorDataManager()
    X = data_manager.load_cardiac_mri()
    output_path = data_manager.output_path
    print(data_manager.output_path)
    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)

    # Hyperspectral
    data_manager = TensorDataManager()
    X = data_manager.load_hyperspectral()
    output_path = data_manager.output_path
    print(data_manager.output_path)
    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)

    # Traffic
    data_manager = TensorDataManager()
    X = data_manager.load_traffic()
    output_path = data_manager.output_path
    print(data_manager.output_path)
    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)

    return


    data_manager = TensorDataManager()
    #X = data_manager.generate_random_normal(shape=(50, 50, 50))
    X = data_manager.generate_random_normal(shape=(100, 100, 100))

    #X = data_manager.generate_random_cp(shape=(50, 50, 50), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=64)

    #X = data_manager.generate_random_tucker(shape=(50, 50, 50), rank=(4, 4, 4))
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
    epsilon = 0.01

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
    return

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

    run_cp_completion_sweep(X, output_path)
    run_lifted_cp_completion_sweep(X, output_path)


main()

