import numpy as np
import random
import time

import matplotlib.pyplot as plt
import tensorly as tl

from scipy.linalg import solve, sqrtm

def comp_lev_scores(A, reg=0):
    """
    Compute the ridge leverage scores for matrix A with regularization parameter reg.

    Parameters:
        A (numpy.ndarray): Input matrix of shape (m, n).
        reg (float): Regularization parameter (must be non-negative).

    Returns:
        numpy.ndarray: Ridge leverage scores, a 1D array of length m.
    """
    if reg < 0:
        raise ValueError("Regularization parameter 'reg' must be non-negative.")
    
    # Compute the regularized Gram matrix
    AtA = A.T @ A
    n = A.shape[1]
    reg_matrix = reg * np.eye(n)
    regularized_AtA = AtA + reg_matrix

    # Compute the inverse of the regularized Gram matrix efficiently
    regularized_AtA_inv = solve(regularized_AtA, np.eye(n), assume_a='sym')

    # Compute the leverage scores
    ridge_leverage_scores = np.sum((A @ regularized_AtA_inv) * A, axis=1)

    return ridge_leverage_scores

def sample_based_on_dist(probs, m):
    """
    Samples m integers from 1 to len(probs) based on the probability distribution defined by probs.

    Parameters:
        probs (numpy.ndarray): Array of probabilities of size n. Must sum to 1.
        m (int): Number of samples to draw.

    Returns:
        numpy.ndarray: Array of m sampled integers (0-indexed).
    """
    if m <= 0:
        raise ValueError("The number of samples 'm' must be a positive integer.")
    
    if not np.isclose(np.sum(probs), 1.0):
        raise ValueError("The probability array 'probs' must sum to 1.")
    
    if np.any(probs < 0):
        raise ValueError("The probability array 'probs' must contain non-negative values.")
    
    n = len(probs)
    samples = np.random.choice(np.arange(0, n), size=m, p=probs)
    
    return samples


def khatri_rao_prod(A, B):
    """
    Efficiently computes the Khatri-Rao product of two matrices A and B with respect to rows.
    
    Parameters:
        A (numpy.ndarray): Matrix of shape (m, n).
        B (numpy.ndarray): Matrix of shape (m, p).
    
    Returns:
        numpy.ndarray: The Khatri-Rao product of A and B, a matrix of shape (m, n * p).
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("Matrices A and B must have the same number of rows.")
    
    # Compute the Khatri-Rao product for all rows at once using broadcasting
    m, n = A.shape
    _, p = B.shape
    
    # Reshape A and B for broadcasting
    A_reshaped = A[:, :, np.newaxis]  # shape (m, n, 1)
    B_reshaped = B[:, np.newaxis, :]  # shape (m, 1, p)
    
    # Perform broadcasting and element-wise multiplication
    C = A_reshaped * B_reshaped  # shape (m, n, p)
    
    # Reshape C into a matrix of shape (m, n * p)
    return C.reshape(m, n * p)

def random_mat_exp(n, d, percentages):

	# Generate two random matrices
	mat_1 = np.random.rand(n, d)
	mat_2 = np.random.rand(n, d)

	# Compute the Kronecker product
	M = np.kron(mat_1, mat_2)
	x = np.random.normal(loc=0, scale=5, size=d*d)
	b = M @ x + np.random.normal(loc=0, scale=0.1, size=n*n)

	dir_time = np.zeros((len(percentages)))
	lift_time = np.zeros((len(percentages)))
	lev_time = np.zeros((len(percentages)))
	dir_err = np.zeros((len(percentages)))
	lift_err = np.zeros((len(percentages)))
	lev_err = np.zeros((len(percentages)))
	dir_err_test = np.zeros((len(percentages)))
	lift_err_test = np.zeros((len(percentages)))
	lev_err_test = np.zeros((len(percentages)))
	lift_convergence = np.zeros((len(percentages)))
	lev_convergence = np.zeros((len(percentages)))

	counter = 0
	for p in (percentages*n*n/100):
		inds = random.sample(range(n*n), int(p))
		start = time.time()
		M = np.kron(mat_1, mat_2)
		x_dir = np.linalg.lstsq(M[inds], b[inds])[0]
		end = time.time()
		dir_time[counter] = end-start
		print("direct method time: " + str(end-start))
		start = time.time()
		x_lift = np.zeros((d*d))
		bhat = np.zeros((n*n))
		k = 0
		print("=========================")
		while True:
			k = k + 1
			N1 = np.linalg.inv(mat_1.T @ mat_1) @ mat_1.T
			N2 = np.linalg.inv(mat_2.T @ mat_2) @ mat_2.T
			btil = (mat_2 @ x_lift.reshape(d,d).T @ mat_1.T).T.reshape(n*n)
			bhat[inds] = b[inds] - btil[inds]
			x_lift_2 = (N2 @ bhat.reshape(n,n).T @ N1.T).T.reshape(d*d)
			if k % 2 == 1:
				m = np.linalg.norm(x_lift_2)
				if np.linalg.norm(x_lift_2) < 0.1:
					break
				x_lift = x_lift + x_lift_2
			elif k % 2 == 0:
				alpha = np.linalg.norm(x_lift_2) / m
				x_lift = x_lift + x_lift_2 / (1-alpha)

		end = time.time()
		lift_time[counter] = end-start
		lift_convergence[counter] = k
		print("lifting convergence: " + str(k))
		print("lifting method time: " + str(end-start))

		start = time.time()
		b_tmp = np.zeros((n*n))
		b_tmp[inds] = b[inds]
		x_lev = np.zeros((d*d))
		k = 0
		print("=========================")
		lev1 = comp_lev_scores(mat_1)
		lev2 = comp_lev_scores(mat_2)
		p1 = lev1/sum(lev1)
		p2 = lev2/sum(lev2)
		s = 200*d*d*(1+int(p/(n*n)*2))
		print(1+int(p/(n*n)*2))
		while True:
			k = k + 1
			ind1 = sample_based_on_dist(p1, s)
			ind2 = sample_based_on_dist(p2, s)
			k_inds = ind1 * n + ind2
			w = 1/np.sqrt(p1[ind1]*p2[ind2]*s)
			w_reshaped = w[:, np.newaxis]
			Z = w_reshaped * khatri_rao_prod(mat_1[ind1], mat_2[ind2])
			btil = Z @ x_lev
			is_obs = (b_tmp[k_inds] != 0)
			bhat = (1-is_obs) * btil + (w * is_obs) * b_tmp[k_inds]
			temp_sol = np.linalg.lstsq(Z, bhat)
			x_lev_2 = temp_sol[0]
			if k % 2 == 1:
				m = np.linalg.norm(x_lev - x_lev_2)
				if np.linalg.norm(x_lev - x_lev_2) < 0.1:
					break
				x_lev = x_lev_2
			elif k % 2 == 0:
				alpha = np.linalg.norm(x_lev - x_lev_2) / m
				x_lev = x_lev + (x_lev_2 - x_lev) / (1-alpha)

		end = time.time()
		lev_time[counter] = end-start
		lev_convergence[counter] = k
		print("leverage convergence: " + str(k))
		print("leverage method time: " + str(end-start))


		dir_err[counter] = np.linalg.norm(M[inds] @ x_dir - b[inds])**2 / (n*n)
		lift_err[counter] = np.linalg.norm(M[inds] @ x_lift - b[inds])**2 / (n*n)
		lev_err[counter] = np.linalg.norm(M[inds] @ x_lev - b[inds])**2 / (n*n)
		dir_err_test[counter] = np.linalg.norm(M @ x_dir - b)**2 / (n*n)
		lift_err_test[counter] = np.linalg.norm(M @ x_lift - b)**2 / (n*n)
		lev_err_test[counter] = np.linalg.norm(M @ x_lev - b)**2 / (n*n)
		print(dir_err_test[counter])
		print(lift_err_test[counter])
		print(lev_err_test[counter])
		counter = counter + 1

	return dir_time, lift_time, lev_time, dir_err, lift_err, lev_err,\
		dir_err_test, lift_err_test, lev_err_test, lift_convergence, lev_convergence

def run_exp(n, d, seed):
	random.seed(seed)
	np.random.seed(seed)

	num_trial = 10
	percentages = np.arange(1/10,3/4,1/20) * 100
	
	dir_time = np.zeros((num_trial, len(percentages)))
	lift_time = np.zeros((num_trial, len(percentages)))
	lev_time = np.zeros((num_trial, len(percentages)))
	dir_err = np.zeros((num_trial, len(percentages)))
	lift_err = np.zeros((num_trial, len(percentages)))
	lev_err = np.zeros((num_trial, len(percentages)))
	dir_err_test = np.zeros((num_trial, len(percentages)))
	lift_err_test = np.zeros((num_trial, len(percentages)))
	lev_err_test = np.zeros((num_trial, len(percentages)))
	lift_convergence = np.zeros((num_trial, len(percentages)))
	lev_convergence = np.zeros((num_trial, len(percentages)))

	for i in range(num_trial):
		output = random_mat_exp(n, d, percentages = percentages)
		dir_time[i] = output[0]
		lift_time[i] = output[1]
		lev_time[i] = output[2]
		dir_err[i] = output[3]
		lift_err[i] = output[4]
		lev_err[i] = output[5]
		dir_err_test[i] = output[6]
		lift_err_test[i] = output[7]
		lev_err_test[i] = output[8]
		lift_convergence[i] = output[9]
		lev_convergence[i] = output[10]

	filename = f"Kronecker_output_{n}_{d}_{seed}.npz"
	np.savez(filename, dir_time=dir_time, lift_time=lift_time, lev_time=lev_time,\
		dir_err=dir_err, lift_err=lift_err, lev_err=lev_err,\
		dir_err_test=dir_err_test, lift_err_test=lift_err_test, lev_err_test=lev_err_test,\
		lift_convergence=lift_convergence, lev_convergence=lev_convergence, percentages=percentages)

def plot_results(n, d, seed):

	filename = f"Kronecker_output_{n}_{d}_{seed}.npz"

	loaded_data = np.load(filename)

	percentages = loaded_data['percentages']
	p_reshape = 1 / (percentages[np.newaxis, :] / 100)

	dir_time = loaded_data['dir_time']
	lift_time = loaded_data['lift_time']
	lev_time = loaded_data['lev_time']

	dir_err = loaded_data['dir_err'] * p_reshape
	lift_err = loaded_data['lift_err'] * p_reshape
	lev_err = loaded_data['lev_err'] * p_reshape

	dir_err_test = loaded_data['dir_err_test']
	lift_err_test = loaded_data['lift_err_test']
	lev_err_test = loaded_data['lev_err_test']

	dir_t_m = np.mean(dir_time, axis=0)
	dir_t_s = np.std(dir_time, axis=0)
	lift_t_m = np.mean(lift_time, axis=0)
	lift_t_s = np.std(lift_time, axis=0)
	lev_t_m = np.mean(lev_time, axis=0)
	lev_t_s = np.std(lev_time, axis=0)

	dir_err_m = np.mean(dir_err_test, axis=0)
	dir_err_s = np.std(dir_err_test, axis=0)
	lift_err_m = np.mean(lift_err_test, axis=0)
	lift_err_s = np.std(lift_err_test, axis=0)
	lev_err_m = np.mean(lev_err_test, axis=0)
	lev_err_s = np.std(lev_err_test, axis=0)

	f1 = plt.figure(1)
	
	plt.plot(percentages, dir_t_m, label = "Direct", color="blue")
	plt.fill_between(percentages, dir_t_m - dir_t_s, dir_t_m + dir_t_s, color="blue", alpha=0.2)

	plt.plot(percentages, lift_t_m, '--', label = "Lifting", color="green")
	plt.fill_between(percentages, lift_t_m - lift_t_s, lift_t_m + lift_t_s, color="green", alpha=0.2)

	plt.plot(percentages, lev_t_m, '-.', label = "Leverage Score", color="orange")
	plt.fill_between(percentages, lev_t_m - lev_t_s, lev_t_m + lev_t_s, color="orange", alpha=0.2)

	plt.ylabel("Time")
	plt.xlabel("Percentage of given entries")
	plt.legend(loc='upper right')
	plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

	plt.savefig("time_comparison_plot.png", dpi=300, bbox_inches="tight")

	f2 = plt.figure(2)
	
	plt.plot(percentages, dir_err_m, label = "Direct", color="blue")
	plt.fill_between(percentages, dir_err_m - dir_err_s, dir_err_m + dir_err_s, color="blue", alpha=0.2)

	plt.plot(percentages, lift_err_m, '--', label = "Lifting", color="green")
	plt.fill_between(percentages, lift_err_m - lift_err_s, lift_err_m + lift_err_s, color="green", alpha=0.2)

	plt.plot(percentages, lev_err_m, '-.', label = "Leverage Score", color="orange")
	plt.fill_between(percentages, lev_err_m - lev_err_s, lev_err_m + lev_err_s, color="orange", alpha=0.2)
	
	plt.ylabel("MSE")
	plt.xlabel("Percentage of given entries")
	plt.legend(loc='upper right')
	plt.ylim([0.005,0.02])
	plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

	plt.savefig("error_comparison_plot.png", dpi=300, bbox_inches="tight")

	# f3 = plt.figure(3)
	# plt.plot(percentages, lift_convergence, label = "Lifting Method")
	# plt.ylabel("Number of steps to convergence")
	# plt.xlabel("Percentage of given entries")

	#plt.show()

n = 2000
d = 10
seed=12345
#run_exp(n, d, seed)
plot_results(n, d, seed)