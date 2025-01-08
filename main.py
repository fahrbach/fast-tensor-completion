from cp_completion_solvers import *
from tensor_data_manager import TensorDataManager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

import time

"""
TODO:
    - Use CpCompletionOutput data class
    - Cache solve outputs
    - Add verbose option for each solve
"""
def main():
    SEED = 0
    SAMPLE_RATIO = 0.01
    NUM_ITERATIONS = 10

    np.random.seed(SEED)
    colors = mpl.colormaps['tab10'].colors

    data_manager = TensorDataManager()
    #X = data_manager.generate_random_normal(shape=(100, 100, 100))
    X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))
    output_path = data_manager.output_path
    print('X.shape:', X.shape)
    print('X.size:', X.size)
    print(data_manager.output_path)

    # Create train and test data.
    shuffled_indices = np.random.permutation(np.arange(X.size))
    num_train_samples = int(SAMPLE_RATIO * X.size)
    print('sample_ratio:', SAMPLE_RATIO)
    print('num_train_samples:', num_train_samples)
    train_indices = shuffled_indices[:num_train_samples]
    num_test_samples = int(0.1 * X.size)
    test_indices = shuffled_indices[-num_test_samples:]  # Use last 10% of shuffled indices.

    # Rank sweep
    if True:
        for i, rank in enumerate([1, 2, 4, 8, 16]):
            print('solving rank:', rank)

            solve_result = run_cp_completion(X, train_indices, rank, output_path, NUM_ITERATIONS, test_indices)

            train_losses = solve_result.train_losses
            test_losses = solve_result.test_losses
            running_times = solve_result.step_times_seconds
            plt.plot(train_losses, label='rank: ' + str(rank), c=colors[i])
            plt.plot(test_losses, linestyle='dashed', c=colors[i]) 
            print(rank, train_losses[-1], test_losses[-1], np.sum(running_times))

        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('CP completion (sample_ratio: ' + str(SAMPLE_RATIO) + ')')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()

    # Sweep over sample ratio
    if True:
        rank = 4

        sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
        #sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        all_running_times = []
        for i, ratio in enumerate(sample_ratios):
            num_train_samples = int(X.size * ratio)
            print('ratio:', ratio, 'num_train_samples:', num_train_samples)
            train_indices = shuffled_indices[:num_train_samples]

            solve_result = run_cp_completion(X, train_indices, rank, output_path, NUM_ITERATIONS, test_indices)

            train_losses = solve_result.train_losses
            test_losses = solve_result.test_losses
            running_times = solve_result.step_times_seconds
            total_solve_time = np.sum(running_times)
            all_running_times.append(total_solve_time)
            print(rank, train_losses[-1], test_losses[-1], total_solve_time)

            plt.plot(train_losses, label='num_samples: ' + str(ratio), c=colors[i])
            plt.plot(test_losses, linestyle='dashed', c=colors[i])

        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('CP completion (rank=4)')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.show()

        plt.xlabel('sample_ratios')
        plt.ylabel('solve times (s)')
        plt.title('CP completion (rank=4)')
        plt.grid()
        plt.plot(sample_ratios, all_running_times, marker='.')
        plt.show()

main()
