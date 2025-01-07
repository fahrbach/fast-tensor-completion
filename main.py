import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np

import matplotlib as mpl

from scipy.misc import face
from scipy.ndimage import zoom

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

"""
TODO:
- Make train/test index notation better
- Comment `run_cp_completion` function
- Create GitHub repo
- Add timer + preliminary runtime plot
- Add L2 regularization param
"""

def tensor_index_to_vec_index(tensor_index, shape):
    return np.ravel_multi_index(tensor_index, shape)

def vec_index_to_tensor_index(vec_index, shape):
    return np.unravel_index(vec_index, shape)

def solve_least_squares(A, b):
    return np.linalg.pinv(A.T @ A) @ (A.T @ b)

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

def run_cp_completion(X, observation_indices, rank, num_iterations, test_indices=None, seed=0):
    np.random.seed(seed)

    factors = [np.random.normal(0, 1, size=(X.shape[n], rank)) for n in range(X.ndim)]

    loss = cp_loss(factors, X, observation_indices)
    losses = [loss]

    loss_test = cp_loss(factors, X, test_indices)
    losses_test = [loss_test]

    for iteration in range(num_iterations):
        # CP completion (need to partition indices on each dimension?)
        for n in range(X.ndim):
            partitioned_indices = [[] for _ in range(X.shape[n])]
            for vec_index in observation_indices:
                tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
                partitioned_indices[tensor_index[n]].append(vec_index)

            # Solve each row of A^{(n)} independently.
            for i in range(X.shape[n]):
                #print(i, partitioned_indices[i])
                num_samples = len(partitioned_indices[i])
                if num_samples == 0:
                    continue
                design_matrix = np.ones((num_samples, rank))
                #print(design_matrix)
                response = np.zeros(num_samples)
                for j in range(num_samples):
                    vec_index = partitioned_indices[i][j]
                    tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
                    #print(' -', tensor_index)
                    for d in range(X.ndim):
                        if d == n:
                            continue
                        factor_row = factors[d][tensor_index[d]]
                        design_matrix[j] = np.multiply(design_matrix[j], factor_row)
                    response[j] = X[tensor_index]

                #print(design_matrix)
                #print(response)
                x = solve_least_squares(design_matrix, response)
                #print(x)
                factors[n][i] = x
                #print()

            loss = cp_loss(factors, X, observation_indices)
            losses.append(loss)

            loss_test = cp_loss(factors, X, test_indices)
            losses_test.append(loss_test)

    return factors, losses, losses_test

def main():
    SEED = 0
    NUM_OBSERVATIONS = 1000
    NUM_ITERATIONS = 20

    np.random.seed(SEED)
    colors = mpl.colormaps['tab10'].colors

    # Get image tensor
    image = face()
    image = tl.tensor(zoom(face(), (0.3, 0.3, 1)), dtype="float64")

    X = image
    print('X.shape:', X.shape)
    X_vec = tl.tensor_to_vec(X)
    indices = np.random.permutation(np.arange(X.size))
    train_indices = indices[:NUM_OBSERVATIONS]
    test_indices = indices[-10000:]

    print('% observations:', NUM_OBSERVATIONS / X.size)

    # Rank sweep
    for rank in range(1, 4 + 1):
        print('rank:', rank)
        factors, losses, losses_test = run_cp_completion(X, train_indices, rank, NUM_ITERATIONS, test_indices)
        plt.plot(losses, label='rank: ' + str(rank), c=colors[rank-1])
        plt.plot(losses_test, linestyle='dashed', c=colors[rank-1]) 

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('CP completion')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

    # Sweep over # samples
    rank = 4
    for i, ratio in enumerate([0.01, 0.02, 0.03, 0.04, 0.05]):
        num_samples = int(X.size * ratio)
        print('ratio:', ratio, 'num_samples:', num_samples)
        train_indices = indices[:num_samples]
        factors, losses, losses_test = run_cp_completion(X, train_indices, rank, NUM_ITERATIONS, test_indices)

        plt.plot(losses, label='num_samples: ' + str(ratio), c=colors[i])
        plt.plot(losses_test, linestyle='dashed', c=colors[i])

    plt.xlabel('step')
    plt.ylabel('loss')
    plt.title('CP completion (rank=4)')
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()

main()
