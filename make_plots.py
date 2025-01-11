from cp_completion_solvers import *
from tensor_data_manager import TensorDataManager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

import os


def main():
    colors = mpl.colormaps['tab10'].colors

    # Set tensor data.
    data_manager = TensorDataManager()
    #X = data_manager.generate_random_normal(shape=(50, 50, 50))
    X = data_manager.generate_random_cp(shape=(50, 50, 50), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=64)
    #X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))
    #X = data_manager.load_cardiac_mri()
    #X = data_manager.load_hyperspectral()
    print('X.shape:', X.shape)
    print('X.size:', X.size)

    # Set algorithm.
    algorithm = 'cp-completion'
    #algorithm = 'lifted-cp-completion'

    cache_dir = data_manager.output_path
    figure_dir = data_manager.output_path + 'figures/' + algorithm + '/'
    print('cache_dir:', cache_dir)
    print('figure_dir:', figure_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # Full sweep. 
    sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    ranks = [1, 2, 4, 8, 16]

    # Plot 1: sample ratio vs train loss
    for i, rank in enumerate(ranks):
        y_train_losses = []
        for sample_ratio in sample_ratios:
            result = run_cp_completion(X, sample_ratio, rank, cache_dir)
            y_train_losses.append(result.train_rres[-1])
        plt.plot(sample_ratios, y_train_losses, label='rank: ' + str(rank), c=colors[i])

    plt.xlabel('sample ratio')
    plt.ylabel('train loss')
    plt.title('CP completion')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'sample-ratio_vs_train-loss.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 2: sample ratio vs test loss
    for i, rank in enumerate(ranks):
        y_test_losses = []
        for sample_ratio in sample_ratios:
            result = run_cp_completion(X, sample_ratio, rank, cache_dir)
            y_test_losses.append(result.test_rres[-1])
        plt.plot(sample_ratios, y_test_losses, linestyle='dashed', label='rank: ' + str(rank), c=colors[i])

    plt.xlabel('sample ratio')
    plt.ylabel('test loss')
    plt.title('CP completion')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'sample-ratio_vs_test-loss.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 3: sample ratio vs both losses
    for i, rank in enumerate(ranks):
        y_train_losses = []
        y_test_losses = []
        for sample_ratio in sample_ratios:
            result = run_cp_completion(X, sample_ratio, rank, cache_dir)
            y_train_losses.append(result.train_rres[-1])
            y_test_losses.append(result.test_rres[-1])
        plt.plot(sample_ratios, y_train_losses, label='rank: ' + str(rank), c=colors[i])
        plt.plot(sample_ratios, y_test_losses, linestyle='dashed', c=colors[i])

    plt.xlabel('sample ratio')
    plt.ylabel('loss')
    plt.title('CP completion')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'sample-ratio_vs_both-loss.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 4: sample ratio vs solve time 
    for i, rank in enumerate(ranks):
        y_solve_times = []
        for sample_ratio in sample_ratios:
            result = run_cp_completion(X, sample_ratio, rank, cache_dir)
            y_solve_times.append(np.sum(result.step_times_seconds))
        plt.plot(sample_ratios, y_solve_times, label='rank: ' + str(rank), c=colors[i])

    plt.xlabel('sample ratio')
    plt.ylabel('solve time (s)')
    plt.title('CP completion')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'sample-ratio_vs_solve-time.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # -------------------------------------------------------------------------------------------

    # Plot 5: step vs train loss (fixed sample_ratio)
    sample_ratio = 0.1
    for i, rank in enumerate(ranks):
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        plt.plot(result.train_rres, label='rank: ' + str(rank), c=colors[i])

    plt.xlabel('step')
    plt.ylabel('train loss')
    plt.title('CP completion (sample_ratio: ' + str(sample_ratio) + ')')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_train-loss_fixed-sample-ratio.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 6: step vs test loss (fixed sample_ratio)
    sample_ratio = 0.1
    for i, rank in enumerate(ranks):
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        plt.plot(result.test_rres, linestyle='dashed', label='rank: ' + str(rank), c=colors[i])

    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.title('CP completion (sample_ratio: ' + str(sample_ratio) + ')')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_test-loss_fixed-sample-ratio.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 7: step vs both losses (fixed sample_ratio)
    sample_ratio = 0.1
    for i, rank in enumerate(ranks):
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        plt.plot(result.train_rres, label='rank: ' + str(rank), c=colors[i])
        plt.plot(result.test_rres, linestyle='dashed', c=colors[i])

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('CP completion (sample_ratio: ' + str(sample_ratio) + ')')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_both-loss_fixed-sample-ratio.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # -------------------------------------------------------------------------------------------

    # Plot 8: step vs train loss (fixed rank)
    rank = 16
    sample_ratios = [0.02, 0.04, 0.06, 0.08, 0.10]
    for i, sample_ratio in enumerate(sample_ratios):
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        plt.plot(result.train_rres, label='sample_ratio: ' + str(sample_ratio), c=colors[i])

    plt.xlabel('step')
    plt.ylabel('train loss')
    plt.title('CP completion (rank: ' + str(rank) + ')')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_train-loss_fixed-rank.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 9: step vs test loss (fixed rank)
    rank = 16
    sample_ratios = [0.02, 0.04, 0.06, 0.08, 0.10]
    for i, sample_ratio in enumerate(sample_ratios):
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        plt.plot(result.test_rres, linestyle='dashed', label='sample_ratio: ' + str(sample_ratio), c=colors[i])

    plt.xlabel('step')
    plt.ylabel('test loss')
    plt.title('CP completion (rank: ' + str(rank) + ')')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_test-loss_fixed-rank.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 10: step vs both losses (fixed rank)
    rank = 16
    sample_ratios = [0.02, 0.04, 0.06, 0.08, 0.10]
    for i, sample_ratio in enumerate(sample_ratios):
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        plt.plot(result.train_rres, label='sample_ratio: ' + str(sample_ratio), c=colors[i])
        plt.plot(result.test_rres, linestyle='dashed', c=colors[i])

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('CP completion (rank: ' + str(rank) + ')')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_both-loss_fixed-rank.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()


main()

