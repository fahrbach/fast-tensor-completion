import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
import time

import matplotlib as mpl

from scipy.misc import face
from scipy.ndimage import zoom

"""
TODO:
- Make train/test index notation better
- Comment `run_cp_completion` function
"""

# Tensor data methods

def get_image_tensor():
    return tl.tensor(zoom(face(), (0.3, 0.3, 1)), dtype="float64")


def get_random_cp_tensor(shape, rank, random_state=1234):
    return tl.random.random_cp(shape, rank, full=True, random_state=random_state)


# Tensor utils 

def tensor_index_to_vec_index(tensor_index, shape):
    return np.ravel_multi_index(tensor_index, shape)


def vec_index_to_tensor_index(vec_index, shape):
    return np.unravel_index(vec_index, shape)


# CP completion

def cp_loss(factors, X, observation_indices):
    rank = factors[0].shape[1]
    num_samples = len(observation_indices)
    assert num_samples != 0

    loss = 0
    for vec_index in observation_indices:
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


def solve_least_squares(A, b, l2_regularization_strength=0.0):
    d = A.shape[1]
    return np.linalg.pinv(A.T @ A + l2_regularization_strength * np.identity(d)) @ (A.T @ b)


def run_cp_completion(X, observation_indices, rank, num_iterations, test_indices=None, seed=0):
    start_time = time.time()

    np.random.seed(seed)
    factors = [np.random.normal(0, 1, size=(X.shape[n], rank)) for n in range(X.ndim)]

    loss = cp_loss(factors, X, observation_indices)
    losses = [loss]

    loss_test = cp_loss(factors, X, test_indices)
    losses_test = [loss_test]

    # Precompute how indices are partitioned.
    # CP completion (need to partition indices on each dimension?)
    partitioned_indices = [[] for _ in range(X.ndim)]
    for n in range(X.ndim):
        partitioned_indices[n] = [[] for _ in range(X.shape[n])]
        for vec_index in observation_indices:
            tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
            partitioned_indices[n][tensor_index[n]].append(vec_index)

    for iteration in range(num_iterations):
        for n in range(X.ndim):
            # Solve each row of A^{(n)} independently.
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

            loss = cp_loss(factors, X, observation_indices)
            losses.append(loss)

            loss_test = cp_loss(factors, X, test_indices)
            losses_test.append(loss_test)

    end_time = time.time()
    solve_time = end_time - start_time

    return factors, losses, losses_test, solve_time


def main():
    SEED = 0
    SAMPLE_RATIO = 0.01
    NUM_ITERATIONS = 10

    np.random.seed(SEED)
    colors = mpl.colormaps['tab10'].colors

    #X = get_image_tensor()
    X = get_random_cp_tensor(shape=(100, 100, 100), rank=16)
    print('X.shape:', X.shape)
    print('X.size:', X.size)

    # Create test and train dataset.
    shuffled_indices = np.random.permutation(np.arange(X.size))
    num_samples_train = int(SAMPLE_RATIO * X.size)
    print('sample_ratio:', SAMPLE_RATIO)
    print('num_samples:', num_samples_train)
    train_indices = shuffled_indices[:num_samples_train]
    num_samples_test = int(0.1 * X.size)
    test_indices = shuffled_indices[-num_samples_test:]  # Use last 10% of shuffled indices.

    # Rank sweep
    if False:
        for i, rank in enumerate([1, 2, 4, 8, 16]):
            print('solving rank:', rank)
            factors, losses, losses_test, solve_time = run_cp_completion(X, train_indices, rank, NUM_ITERATIONS, test_indices)
            plt.plot(losses, label='rank: ' + str(rank), c=colors[i])
            plt.plot(losses_test, linestyle='dashed', c=colors[i]) 
            print(rank, losses[-1], losses_test[-1], solve_time)

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

        #sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05]
        sample_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        running_times = []
        for i, ratio in enumerate(sample_ratios):
            num_samples = int(X.size * ratio)
            print('ratio:', ratio, 'num_samples:', num_samples)
            train_indices = shuffled_indices[:num_samples]
            factors, losses, losses_test, solve_time = run_cp_completion(X, train_indices, rank, NUM_ITERATIONS, test_indices)
            running_times.append(solve_time)
            print(rank, losses[-1], losses_test[-1], solve_time)

            plt.plot(losses, label='num_samples: ' + str(ratio), c=colors[i])
            plt.plot(losses_test, linestyle='dashed', c=colors[i])

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
        plt.plot(sample_ratios, running_times, marker='.')
        plt.show()

main()
