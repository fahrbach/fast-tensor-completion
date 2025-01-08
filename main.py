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
    - Refactor into [tensor_data_handler.py, tensor_utils.py, cp_completion_solvers.py]
"""

# Tensor utils 

def tensor_index_to_vec_index(tensor_index, shape):
    return np.ravel_multi_index(tensor_index, shape)


def vec_index_to_tensor_index(vec_index, shape):
    return np.unravel_index(vec_index, shape)


# CP completion

# TODO: Make faster. About 90+% of experiment time is spent in this function.
def cp_loss(factors, X, indices):
    """
    Returns the mean-squared error of the CP decomposition over `indices`.
    """
    rank = factors[0].shape[1]
    num_samples = len(indices)
    assert num_samples != 0

    loss = 0.0
    for vec_index in indices:
        tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
        y = X[tensor_index]
        
        tmp = np.ones(rank)
        for n in range(X.ndim):
            row = factors[n][tensor_index[n]]
            tmp = np.multiply(tmp, row)
        y_hat = np.sum(tmp)

        loss += (y - y_hat)**2
    loss /= num_samples
    return loss


def solve_least_squares(A, b, l2_regularization=0.0):
    d = A.shape[1]
    return np.linalg.pinv(A.T @ A + l2_regularization * np.identity(d)) @ (A.T @ b)


def run_cp_completion(X, train_indices, rank, num_iterations, test_indices=None, seed=0):
    train_losses = []
    test_losses = []
    running_times = []  # Ignores loss computations.

    # Initialization.
    init_start_time = time.time()

    np.random.seed(seed)
    factors = [np.random.normal(0, 1, size=(X.shape[n], rank)) for n in range(X.ndim)]

    # Precompute how indices are partitioned.
    partitioned_indices = [[] for _ in range(X.ndim)]
    for n in range(X.ndim):
        partitioned_indices[n] = [[] for _ in range(X.shape[n])]
        for vec_index in train_indices:
            tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
            partitioned_indices[n][tensor_index[n]].append(vec_index)

    init_duration = time.time() - init_start_time
    running_times.append(init_duration)

    loss = cp_loss(factors, X, train_indices)
    train_losses = [loss]
    loss = cp_loss(factors, X, test_indices)
    test_losses = [loss]

    for iteration in range(num_iterations):
        for n in range(X.ndim):
            start_step_time = time.time()
            # Update each row of A^{(n)} independently.
            for i in range(X.shape[n]):
                num_samples = len(partitioned_indices[n][i])
                if num_samples == 0:
                    continue
                design_matrix = np.ones((num_samples, rank))
                response = np.zeros(num_samples)
                for j in range(num_samples):
                    vec_index = partitioned_indices[n][i][j]
                    tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
                    for d in range(X.ndim):
                        if d == n:
                            continue
                        factor_row = factors[d][tensor_index[d]]
                        design_matrix[j] = np.multiply(design_matrix[j], factor_row)
                    response[j] = X[tensor_index]

                x = solve_least_squares(design_matrix, response)
                factors[n][i] = x

            step_duration = time.time() - start_step_time
            running_times.append(step_duration)

            loss = cp_loss(factors, X, train_indices)
            train_losses.append(loss)
            loss = cp_loss(factors, X, test_indices)
            test_losses.append(loss)

    assert len(train_losses) == len(test_losses)
    assert len(train_losses) == len(running_times)
    return factors, train_losses, test_losses, running_times


def main():
    SEED = 0
    SAMPLE_RATIO = 0.01
    NUM_ITERATIONS = 10

    np.random.seed(SEED)
    colors = mpl.colormaps['tab10'].colors

    data_manager = TensorDataManager()
    X = data_manager.generate_random_normal(shape=(100, 100, 100))
    #X = data_manager.generate_random_cp(shape=(100, 100, 100), rank=16)
    #X = data_manager.generate_random_tucker(shape=(100, 100, 100), rank=(4, 4, 4))
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
            factors, train_losses, test_losses, running_times = run_cp_completion(X, train_indices, rank, NUM_ITERATIONS, test_indices)
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
            factors, train_losses, test_losses, running_times = run_cp_completion(X, train_indices, rank, NUM_ITERATIONS, test_indices)
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
