import numpy as np

import dataclasses
import datetime
import os
import time


@dataclasses.dataclass
class CpCompletionSolveResult:
    date: str = datetime.datetime.now().isoformat()

    sample_ratio: float = None
    rank: int = None
    seed: int = None

    # Instance-level info
    input_shape: list[int] = None
    num_train_samples: int = None
    num_test_samples: int = None

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


# TODO(fahrbach): Make faster. 90+% of solve time is spent in this function.
# TODO(fahrbach): Decide if we want to include regularization in CP loss.
def compute_cp_loss(factors, X, indices):
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


def get_train_and_test_data(X, sample_ratio):
    DATA_SEED = 0
    TEST_RATIO = 0.1  # Use last 10% of shuffled indices.
    np.random.seed(DATA_SEED)
    shuffled_indices = np.random.permutation(np.arange(X.size))
    num_train_samples = int(sample_ratio * X.size)
    print('sample_ratio:', sample_ratio)
    print('num_train_samples:', num_train_samples)
    train_indices = shuffled_indices[:num_train_samples]
    num_test_samples = int(TEST_RATIO * X.size)
    test_indices = shuffled_indices[-num_test_samples:]  
    return train_indices, test_indices

"""
TODO:
    - Change arguments to run_cp_completion(X, sample_ratio, rank, output_path, trial)
      * annotate result file w/ {train, test} sample ratio
"""
def run_cp_completion(X, sample_ratio, rank, output_path, num_iterations, seed=0):
    assert output_path[-1] == '/'

    # Check if the solve result has been cached.
    output_path += 'cp-completion/'
    filename = output_path
    filename += 'sample_ratio-' + str(sample_ratio) + '_'
    filename += 'rank-' + str(rank) + '_'
    filename += 'seed-' + str(seed)
    filename += '.txt'

    if os.path.exists(filename):
        print('cache hit for:', filename)
        solve_result = read_cp_completion_solve_result_from_file(filename)
        print(solve_result)
        return solve_result

    train_indices, test_indices = get_train_and_test_data(X, sample_ratio)

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

    loss = compute_cp_loss(factors, X, train_indices)
    train_losses = [loss]
    loss = compute_cp_loss(factors, X, test_indices)
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

            loss = compute_cp_loss(factors, X, train_indices)
            train_losses.append(loss)
            loss = compute_cp_loss(factors, X, test_indices)
            test_losses.append(loss)

    assert len(train_losses) == len(test_losses)
    assert len(train_losses) == len(running_times)

    solve_result = CpCompletionSolveResult()

    solve_result.sample_ratio = sample_ratio
    solve_result.rank = rank
    solve_result.seed = seed

    solve_result.train_losses = train_losses
    solve_result.test_losses = test_losses
    solve_result.step_times_seconds = running_times

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_dataclass_to_file(solve_result, filename)
    return solve_result

