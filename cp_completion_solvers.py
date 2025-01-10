import numpy as np
import tensorly as tl

import dataclasses
import datetime
import os
import time

# Globals
USE_CACHING = False
L2_REGULARIZATION_STRENGTH = 0 #1e-6
NUM_ITERATIONS = 10


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

    train_losses: list[float] = None
    test_losses: list[float] = None
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


# TODO(fahrbach): Make faster. 90+% of solve time is spent in this function.
def compute_cp_loss(factors, X, indices):
    """
    Returns the mean-squared error of the CP decomposition over `indices`.
    """
    num_samples = len(indices)
    assert num_samples != 0

    # Faster numpy compuation if evaluating on the entire tensor `X`.
    if num_samples == X.size:
        y_vec = tl.base.tensor_to_vec(X)
        y_hat_vec = cp_factors_to_cp_vec(factors)
        assert len(y_vec) == len(y_hat_vec)
        tmp = np.sum((y_vec - y_hat_vec)**2)
        mse = tmp / num_samples
        rre = tmp / np.sum(y_vec**2)
        return rre

    X_vec = tl.base.tensor_to_vec(X)
    X_hat_vec = cp_factors_to_cp_vec(factors)
    y_vec = X_vec[indices]
    y_hat_vec = X_hat_vec[indices]
    tmp = np.sum((y_vec - y_hat_vec)**2)
    mse = tmp / num_samples
    rre = tmp / np.sum(y_vec**2)
    return rre


def solve_least_squares(A, b, l2_regularization):
    d = A.shape[1]
    return np.linalg.pinv(A.T @ A + l2_regularization * np.identity(d)) @ (A.T @ b)


def get_train_and_test_data(X, sample_ratio):
    DATA_SEED = 0
    TEST_RATIO = 1.0 # 0.1  # Use last 10% of shuffled indices.
    np.random.seed(DATA_SEED)
    shuffled_indices = np.random.permutation(np.arange(X.size))
    num_train_samples = int(sample_ratio * X.size)
    print('sample_ratio:', sample_ratio)
    print('num_train_samples:', num_train_samples)
    train_indices = shuffled_indices[:num_train_samples]
    num_test_samples = int(TEST_RATIO * X.size)
    test_indices = shuffled_indices[-num_test_samples:]  
    return train_indices, test_indices


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

    train_indices, test_indices = get_train_and_test_data(X, sample_ratio)

    train_losses = []
    test_losses = []
    running_times = []  # Ignores loss computations.

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
    running_times.append(init_duration)

    loss = compute_cp_loss(factors, X, train_indices)
    train_losses = [loss]
    loss = compute_cp_loss(factors, X, test_indices)
    test_losses = [loss]
 
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
            running_times.append(step_duration)

            loss = compute_cp_loss(factors, X, train_indices)
            train_losses.append(loss)
            print(' - (iteration, n):', (iteration, n), '->', loss)
            loss = compute_cp_loss(factors, X, test_indices)
            test_losses.append(loss)

    assert len(train_losses) == len(test_losses)
    assert len(train_losses) == len(running_times)

    # Construct output
    result = CpCompletionSolveResult()

    result.sample_ratio = sample_ratio
    result.rank = rank
    result.seed = seed
    result.l2_regularization_strength = L2_REGULARIZATION_STRENGTH

    result.input_shape = X.shape
    result.input_size = X.size
    result.num_train_samples = len(train_indices)
    result.num_test_samples = len(train_indices)

    result.num_iterations = NUM_ITERATIONS

    result.train_losses = train_losses
    result.test_losses = test_losses
    result.step_times_seconds = running_times

    if USE_CACHING:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        write_dataclass_to_file(result, filename)
    return result


def run_lifted_cp_completion(X, sample_ratio, rank, output_path, seed=0):
    NUM_RICHARDSON_ITERATIONS = 10

    assert output_path[-1] == '/'

    # Check if the solve result has been cached.
    output_path += 'lifted-cp-completion/'
    filename = output_path
    filename += 'sample_ratio-' + str(sample_ratio) + '_'
    filename += 'rank-' + str(rank) + '_'
    filename += 'seed-' + str(seed)
    filename += '.txt'

    if USE_CACHING and os.path.exists(filename):
        result = read_cp_completion_solve_result_from_file(filename)
        assert result.num_iterations >= NUM_ITERATIONS
        return result

    train_indices, test_indices = get_train_and_test_data(X, sample_ratio)

    train_losses = []
    test_losses = []
    running_times = []  # Ignores loss computations.

    # Initialization.
    init_start_time = time.time()

    np.random.seed(seed)
    factors = [np.random.normal(0, 0.1, size=(X.shape[n], rank)) for n in range(X.ndim)]
    for i, factor in enumerate(factors):
        print('factor', i, factor.shape)

    X_vec = tl.base.tensor_to_vec(X)
    y_train = X_vec[train_indices]
    del X_vec

    init_duration = time.time() - init_start_time
    running_times.append(init_duration)

    train_loss = compute_cp_loss(factors, X, train_indices)
    train_losses = [train_loss]
    test_loss = compute_cp_loss(factors, X, test_indices)
    test_losses = [test_loss]
    print('0: train_loss:', train_loss, 'test_loss:', test_loss)

    # Alternating least squares
    for iteration in range(NUM_ITERATIONS):
        for n in range(X.ndim):
            start_step_time = time.time()

            #print(' # dimension:', n)

            print(' - creating design_matrix...')
            design_matrix = tl.tenalg.khatri_rao(factors, skip_matrix=n)
            print(' - computing gram matrix...')
            # TODO: Take advantage of structure here.
            d = design_matrix.shape[1]
            tmp = np.linalg.pinv(design_matrix.T @ design_matrix + L2_REGULARIZATION_STRENGTH * np.identity(d))
            #print(' * design_matrix.shape', design_matrix.shape)
 
            for j in range(NUM_RICHARDSON_ITERATIONS):
                print(' - richardson step:', j)
                print('   * computing y_vec...')
                y_vec = cp_factors_to_cp_vec(factors)
                y_vec[train_indices] = y_train
                #print(' * y_vec.shape:', y_vec.shape)
                print('   * computing Y...')
                Y = tl.base.vec_to_tensor(y_vec, X.shape)
                #print(' * Y.shape:', Y.shape)
                print('   * computing Y_unfolded_n ...')
                Y_unfolded_n = tl.base.unfold(Y, n)
                #print(' * Y_unfolded_n.shape:', Y_unfolded_n.shape)
                #print(' * factors[n].shape:', factors[n].shape)

                # Structured solve step?
                print('   * computing tmp2 ...')
                tmp2 = design_matrix.T @ Y_unfolded_n.T
                print('   * computing sol...')
                sol = tmp @ tmp2
                #print(' *', tmp.shape, tmp2.shape, sol.shape, factors[n].shape)
                factors[n] = sol.T

            step_duration = time.time() - start_step_time
            running_times.append(step_duration)

            loss = compute_cp_loss(factors, X, train_indices)
            train_losses.append(loss)
            print(' - (iteration, n):', (iteration, n), '->', loss)
            loss = compute_cp_loss(factors, X, test_indices)
            test_losses.append(loss)

    assert len(train_losses) == len(test_losses)
    assert len(train_losses) == len(running_times)

    # Construct output
    result = CpCompletionSolveResult()

    result.sample_ratio = sample_ratio
    result.rank = rank
    result.seed = seed
    result.l2_regularization_strength = L2_REGULARIZATION_STRENGTH

    result.input_shape = X.shape
    result.input_size = X.size
    result.num_train_samples = len(train_indices)
    result.num_test_samples = len(train_indices)

    result.num_iterations = NUM_ITERATIONS

    result.train_losses = train_losses
    result.test_losses = test_losses
    result.step_times_seconds = running_times

    if USE_CACHING:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        write_dataclass_to_file(result, filename)
    return result

