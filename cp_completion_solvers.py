import numpy as np
import tensorly as tl

import dataclasses
import datetime
import os
import time

# Globals
USE_CACHING = True
L2_REGULARIZATION_STRENGTH = 0 # 1e-3
NUM_ITERATIONS = 10
MAX_NUM_RICHARDSON_ITERATIONS = 1024


@dataclasses.dataclass
class CpCompletionSolveResult:
    date: str = datetime.datetime.now().isoformat()

    sample_ratio: float = None
    rank: int = None
    seed: int = None
    l2_regularization_strength: float = None

    # Instance-level info.
    input_shape: list[int] = None
    input_size: int = None
    num_train_samples: int = None
    num_test_samples: int = None

    # Solver settings.
    num_iterations: int = None
    max_num_richardson_iterations: int = None
    num_richardson_iterations: list[int] = None

    train_mses: list[float] = None
    train_rres: list[float] = None
    test_mses: list[float] = None
    test_rres: list[float] = None
    step_times_seconds: list[float] = None


def write_dataclass_to_file(dataclass, filename):
    """
    Writes `dataclass` (key, values) to `filename`.
    """
    with open(filename, 'w') as f:
        for field in dataclasses.fields(dataclass):
            f.write(str(field.name) + ' ')
            f.write(str(getattr(dataclass, field.name)) + '\n')


def read_cp_completion_solve_result_from_file(filename):
    """
    Reads `filename` and constructs the `CpCompletionSolveResult` data.
    """
    assert os.path.exists(filename)
    solve_result = CpCompletionSolveResult()
    with open(filename, 'r') as f:
        lines = f.readlines()
        assert len(lines) == len(dataclasses.fields(solve_result))
        for line in lines:
            line = line.strip()
            tokens = line.split()
            assert len(tokens) >= 2
            key = tokens[0]
            value_str = ' '.join(tokens[1:])
            value = value_str
            # Update numeric values to not be strings.
            if key not in ['date']:
                value = eval(value_str)
            setattr(solve_result, key, value)
    return solve_result


def tensor_index_to_vec_index(tensor_index, shape):
    return np.ravel_multi_index(tensor_index, shape)


def vec_index_to_tensor_index(vec_index, shape):
    return np.unravel_index(vec_index, shape)


def cp_factors_to_cp_vec(factors):
    return tl.tenalg.khatri_rao(factors).sum(axis=1)


def compute_cp_errors(factors, x_vec, train_indices):
    x_hat_vec = cp_factors_to_cp_vec(factors)

    # Compute train errors
    y_vec = x_vec[train_indices]
    y_hat_vec = x_hat_vec[train_indices]
    num = np.sum((y_vec - y_hat_vec)**2)
    train_mse = num / y_vec.size
    train_rre = num / np.sum(y_vec**2)

    # Compute test errors
    num = np.sum((x_vec - x_hat_vec)**2)
    test_mse = num / x_vec.size
    test_rre = num / np.sum(x_vec**2)

    return train_mse, train_rre, test_mse, test_rre


def solve_least_squares(A, b, l2_regularization):
    d = A.shape[1]
    return np.linalg.pinv(A.T @ A + l2_regularization * np.identity(d)) @ (A.T @ b)


def get_train_indices(X, sample_ratio):
    DATA_SEED = 0
    np.random.seed(DATA_SEED)
    shuffled_indices = np.random.permutation(np.arange(X.size))
    num_train_samples = int(sample_ratio * X.size)
    print('sample_ratio:', sample_ratio)
    print('num_train_samples:', num_train_samples)
    return shuffled_indices[:num_train_samples]


def run_cp_completion(X, sample_ratio, rank, output_path, seed=0):
    assert output_path[-1] == '/'

    # Check if the solve result has been cached.
    output_path += 'cp-completion/'
    filename = output_path
    filename += 'sample_ratio-' + str(sample_ratio) + '_'
    filename += 'rank-' + str(rank) + '_'
    filename += 'seed-' + str(seed)
    filename += '.txt'

    if USE_CACHING and os.path.exists(filename):
        result = read_cp_completion_solve_result_from_file(filename)
        assert result.num_iterations >= NUM_ITERATIONS
        return result

    train_indices = get_train_indices(X, sample_ratio)

    train_mses = []
    train_rres = []
    test_mses = []
    test_rres = []
    step_times = []  # Ignores loss computations.
    x_vec = tl.base.tensor_to_vec(X)

    # Initialization.
    init_start_time = time.time()

    np.random.seed(seed)
    factors = [np.random.normal(0, 0.1, size=(X.shape[n], rank)) for n in range(X.ndim)]

    # Precompute how indices are partitioned.
    partitioned_indices = [[] for _ in range(X.ndim)]
    for n in range(X.ndim):
        partitioned_indices[n] = [[] for _ in range(X.shape[n])]
        for vec_index in train_indices:
            tensor_index = vec_index_to_tensor_index(vec_index, X.shape)
            partitioned_indices[n][tensor_index[n]].append(vec_index)

    init_duration = time.time() - init_start_time
    step_times.append(init_duration)
    
    train_mse, train_rre, test_mse, test_rre = compute_cp_errors(factors, x_vec, train_indices)
    train_mses.append(train_mse)
    train_rres.append(train_rre)
    test_mses.append(test_mse)
    test_rres.append(test_rre)

    # Alternating least squares
    for iteration in range(NUM_ITERATIONS):
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

                x = solve_least_squares(design_matrix, response, L2_REGULARIZATION_STRENGTH)
                factors[n][i] = x

            step_duration = time.time() - start_step_time
            step_times.append(step_duration)

            train_mse, train_rre, test_mse, test_rre = compute_cp_errors(factors, x_vec, train_indices)
            train_mses.append(train_mse)
            train_rres.append(train_rre)
            test_mses.append(test_mse)
            test_rres.append(test_rre)
            print(' - (iteration, n):', (iteration, n), '->', train_rre, test_rre)

    # Construct output
    result = CpCompletionSolveResult()

    result.sample_ratio = sample_ratio
    result.rank = rank
    result.seed = seed
    result.l2_regularization_strength = L2_REGULARIZATION_STRENGTH

    result.input_shape = X.shape
    result.input_size = X.size
    result.num_train_samples = len(train_indices)
    result.num_test_samples = X.size

    result.num_iterations = NUM_ITERATIONS

    result.train_mses = train_mses
    result.train_rres = train_rres
    result.test_mses = test_mses
    result.test_rres = test_rres
    result.step_times_seconds = step_times

    if USE_CACHING:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        write_dataclass_to_file(result, filename)
    return result


def run_lifted_cp_completion(X, sample_ratio, rank, output_path, seed=0, epsilon=0.1):
    assert output_path[-1] == '/'

    # Check if the solve result has been cached.
    output_path += 'lifted-cp-completion/'
    filename = output_path
    filename += 'sample_ratio-' + str(sample_ratio) + '_'
    filename += 'rank-' + str(rank) + '_'
    filename += 'seed-' + str(seed) + '_'
    filename += 'epsilon-' + str(epsilon)
    filename += '.txt'

    if USE_CACHING and os.path.exists(filename):
        result = read_cp_completion_solve_result_from_file(filename)
        assert result.num_iterations >= NUM_ITERATIONS
        return result

    train_indices = get_train_indices(X, sample_ratio)

    num_richardson_iterations = []
    train_mses = []
    train_rres = []
    test_mses = []
    test_rres = []
    step_times = []  # Ignores loss computations.

    # Initialization.
    init_start_time = time.time()

    np.random.seed(seed)
    factors = [np.random.normal(0, 0.1, size=(X.shape[n], rank)) for n in range(X.ndim)]
    for i, factor in enumerate(factors):
        print('factor', i, factor.shape)

    x_vec = tl.base.tensor_to_vec(X)
    y_train = x_vec[train_indices]

    init_duration = time.time() - init_start_time
    step_times.append(init_duration)

    train_mse, train_rre, test_mse, test_rre = compute_cp_errors(factors, x_vec, train_indices)
    train_mses.append(train_mse)
    train_rres.append(train_rre)
    test_mses.append(test_mse)
    test_rres.append(test_rre)

    # Alternating least squares
    for iteration in range(NUM_ITERATIONS):
        for n in range(X.ndim):
            start_step_time = time.time()

            print(' - creating design_matrix...')
            design_matrix = tl.tenalg.khatri_rao(factors, skip_matrix=n)
            print(' - computing gram matrix...')
            # TODO: Take advantage of structure here.
            d = design_matrix.shape[1]
            tmp = np.linalg.pinv(design_matrix.T @ design_matrix + L2_REGULARIZATION_STRENGTH * np.identity(d))
 
            richardson_rres = []
            for j in range(MAX_NUM_RICHARDSON_ITERATIONS):
                x_hat_vec = cp_factors_to_cp_vec(factors)
                y_hat = x_hat_vec[train_indices]
                rre = np.sum((y_hat - y_train)**2) / np.sum(y_train**2)
                richardson_rres.append(rre)
                if j == 0:
                    ratio = 1
                else:
                    ratio = 1 - richardson_rres[-1] / richardson_rres[-2]
                print('   * richardson step:', j, rre, ratio)
                if ratio < epsilon:
                    break
                x_hat_vec[train_indices] = y_train
                X_hat = tl.base.vec_to_tensor(x_hat_vec, X.shape)
                X_unfolded_n = tl.base.unfold(X_hat, n)

                # Structured solve step?
                tmp2 = design_matrix.T @ X_unfolded_n.T
                sol = tmp @ tmp2
                factors[n] = sol.T
            num_richardson_iterations.append(len(richardson_rres))

            step_duration = time.time() - start_step_time
            step_times.append(step_duration)

            train_mse, train_rre, test_mse, test_rre = compute_cp_errors(factors, x_vec, train_indices)
            train_mses.append(train_mse)
            train_rres.append(train_rre)
            test_mses.append(test_mse)
            test_rres.append(test_rre)
            print(' - (iteration, n):', (iteration, n), '->', train_rre, test_rre)

    # Construct output
    result = CpCompletionSolveResult()

    result.sample_ratio = sample_ratio
    result.rank = rank
    result.seed = seed
    result.l2_regularization_strength = L2_REGULARIZATION_STRENGTH

    result.input_shape = X.shape
    result.input_size = X.size
    result.num_train_samples = len(train_indices)
    result.num_test_samples = X.size

    result.num_iterations = NUM_ITERATIONS
    result.max_num_richardson_iterations = MAX_NUM_RICHARDSON_ITERATIONS
    result.num_richardson_iterations = num_richardson_iterations

    result.train_mses = train_mses
    result.train_rres = train_rres
    result.test_mses = test_mses
    result.test_rres = test_rres
    result.step_times_seconds = step_times

    if USE_CACHING:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        write_dataclass_to_file(result, filename)
    return result

