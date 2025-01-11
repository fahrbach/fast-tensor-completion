from cp_completion_solvers import *
from tensor_data_manager import TensorDataManager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

import os


def plot_cp_completion_sweep(X, cache_dir):
    colors = mpl.colormaps['tab10'].colors

    # Set algorithm.
    algorithm = 'cp-completion'

    figure_dir = cache_dir + 'figures/' + algorithm + '/'
    print('cache_dir:', cache_dir)
    print('figure_dir:', figure_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    # Full sweep. 
    sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    #sample_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
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
    ranks = [1, 2, 4, 8, 16]
    sample_ratio = 0.10

    # Plot 5: step vs train loss (fixed sample_ratio)
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
    rank = 16
    sample_ratios = [0.02, 0.04, 0.06, 0.08, 0.10]
    #sample_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Plot 8: step vs train loss (fixed rank)
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


def plot_lifted_comparison(X, cache_dir):
    colors = mpl.colormaps['tab10'].colors

    figure_dir = cache_dir + 'figures/lifted-comparison/'
    print('cache_dir:', cache_dir)
    print('figure_dir:', figure_dir)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    sample_ratio = 0.10
    rank = 16
    epsilons = [1, 0.1, 0.01, 0.001]

    # Plot 1: step vs loss (fixed value of epsilon)

    # Lifted methods
    for i, epsilon in enumerate(epsilons):
        result = run_lifted_cp_completion(X, sample_ratio, rank, cache_dir, epsilon=epsilon)
        plt.plot(result.train_rres, label='eps: ' + str(epsilon), c=colors[i])
        plt.plot(result.test_rres, linestyle='dashed', c=colors[i])

    # Direct method
    result = run_cp_completion(X, sample_ratio, rank, cache_dir)
    plt.plot(result.train_rres, label='direct', c=colors[len(epsilons)])
    plt.plot(result.test_rres, linestyle='dashed', c=colors[len(epsilons)])


    plt.xlabel('step')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title('Lifted CP completion (' + str(sample_ratio) + ', ' + str(rank) + ')')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_both-losses.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()


    # Plot 2: step vs runtime (fixed value of epsilon)
    for i, epsilon in enumerate(epsilons):
        result = run_lifted_cp_completion(X, sample_ratio, rank, cache_dir, epsilon=epsilon)
        plt.plot(result.step_times_seconds, label='eps: ' + str(epsilon), c=colors[i])

    # Direct method
    result = run_cp_completion(X, sample_ratio, rank, cache_dir)
    plt.plot(result.step_times_seconds, label='direct', c=colors[len(epsilons)])

    plt.xlabel('step')
    plt.ylabel('solve time (s)')
    plt.title('Lifted CP completion (' + str(sample_ratio) + ', ' + str(rank) + ')')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_solve-time_1.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

    # Plot 3: sample ratio vs runtime (fixed value of epsilon)
    #sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    sample_ratios = [0.05, 0.10, 0.15, 0.20, 0.25]
    for i, epsilon in enumerate(epsilons):
        y_values = []
        for sample_ratio in sample_ratios:
            result = run_lifted_cp_completion(X, sample_ratio, rank, cache_dir, epsilon=epsilon)
            y_values.append(np.sum(result.step_times_seconds))
        plt.plot(sample_ratios, y_values, label='eps: ' + str(epsilon), c=colors[i], marker='x')

    # Direct method
    y_values = []
    for sample_ratio in sample_ratios:
        result = run_cp_completion(X, sample_ratio, rank, cache_dir)
        y_values.append(np.sum(result.step_times_seconds))
    plt.plot(sample_ratios, y_values, label='direct', c=colors[len(epsilons)], marker='o')

    plt.xlabel('sample ratio')
    plt.ylabel('solve time (s)')
    plt.title('Lifted CP completion (' + str(sample_ratio) + ', ' + str(rank) + ')')
    plt.grid()
    plt.legend()
    filename = figure_dir + 'step_vs_solve-time_2.png'
    plt.savefig(filename, transparent=True, bbox_inches='tight', dpi=256)
    plt.show()

def main():
    print('Making plots...')

    # Set tensor data.
    data_manager = TensorDataManager()
    #X = data_manager.generate_random_normal(shape=(50, 50, 50))
    #X = data_manager.generate_random_normal(shape=(100, 100, 100))

    #X = data_manager.generate_random_cp(shape=(50, 50, 50), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=64)

    X = data_manager.generate_random_tucker(shape=(50, 50, 50), rank=(4, 4, 4))
    #X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))

    #X = data_manager.load_cardiac_mri()
    #X = data_manager.load_hyperspectral()
    print('X.shape:', X.shape)
    print('X.size:', X.size)
    print('data_manager.output_path:', data_manager.output_path)

    plot_cp_completion_sweep(X, data_manager.output_path)
    plot_lifted_comparison(X, data_manager.output_path)

main()

