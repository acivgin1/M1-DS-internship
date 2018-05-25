import os

import numpy as np
from ratings import load_train_test
from matplotlib import pyplot as plt
import pyximport; pyximport.install()
from matrix_factorization import svd_predict, svd_train


def plot_progress(rmse_arr, rmse_t_arr, mae_t_arr):
    t = np.arange(1, len(rmse_arr) + 1)

    plt.figure()
    plt.subplot(311)
    plt.plot(t, rmse_arr)
    plt.plot(t, rmse_t_arr)
    plt.plot(t, mae_t_arr)
    plt.legend(['Training', 'Testing', 'MAE: testing'])
    plt.xlabel('Broj iteracija')
    plt.ylabel('Errors')

    plt.subplot(312)
    plt.plot(t[1:], -np.diff(rmse_arr))
    plt.plot(t[1:], -np.diff(rmse_t_arr))
    plt.plot(t[1:], -np.diff(mae_t_arr))
    plt.legend(['Training', 'Testing', 'MAE: testing'])
    plt.xlabel('Broj iteracija')
    plt.ylabel('Errors first derivatives')

    plt.subplot(313)
    plt.plot(t[2:], -np.diff(rmse_arr, n=2))
    plt.plot(t[2:], -np.diff(rmse_t_arr, n=2))
    plt.plot(t[2:], -np.diff(mae_t_arr, n=2))
    plt.legend(['Training', 'Testing', 'MAE: testing'])
    plt.xlabel('Broj iteracija')
    plt.ylabel('Errors first derivatives')

    plt.show()


def save_svd_params(svd_path, bu, bi, pu, qi):
    np.save('{}/bu.npy'.format(svd_path), bu)
    np.save('{}/bi.npy'.format(svd_path), bi)
    np.save('{}/pu.npy'.format(svd_path), pu)
    np.save('{}/qi.npy'.format(svd_path), qi)


def load_svd_params(svd_path):
    bu = np.load('{}/bu.npy'.format(svd_path))
    bi = np.load('{}/bi.npy'.format(svd_path))
    pu = np.load('{}/pu.npy'.format(svd_path))
    qi = np.load('{}/qi.npy'.format(svd_path))
    return bu, bi, pu, qi


def main():
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    svd_path = '{}/SVD_params'.format(data_path)

    sA_train, sA_test = load_train_test(data_path)

    mu, bu, bi, pu, qi, rmse_arr, rmse_t_arr, mae_t_arr = svd_train(sA_train, sA_test,
                                                                    k_order=100,
                                                                    gamma=0.005,
                                                                    beta=0.02,
                                                                    num_of_iters=2,
                                                                    print_state=True)

    # mu = (sA_train.data.sum()/sA_train.data.shape)[0]
