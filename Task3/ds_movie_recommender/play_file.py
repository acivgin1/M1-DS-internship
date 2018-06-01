import os

import numpy as np
from ratings import load_train_validation_test
import pyximport; pyximport.install()
from svd_clustering import SvdCluster

from movie_recommender import SmartQi

GAMMA_LIST = [0.006, 0.005]
BETA_LIST = [0.06, 0.05, 0.04]


def cross_validation(sA_train, sA_validation, sA_test, svd_path):
    min_rmse_v_arr = 100
    best_pair = []

    for gamma in GAMMA_LIST:
        for beta in BETA_LIST:
            svd_cluster = SvdCluster(k_order=28, gamma=gamma, beta=beta,
                                     num_of_iters=61, verbose=True, svd_path=svd_path)
            svd_cluster.svd_train(sA_train, sA_validation, sA_test, print_step_size=30)
            if np.where(np.diff(svd_cluster.rmse_v_arr) > 0)[0].size > 0:
                print('Error is rising')
                continue
            if svd_cluster.rmse_v_arr[-1] < min_rmse_v_arr:
                min_rmse_v_arr = svd_cluster.rmse_v_arr[-1]
                best_pair = [gamma, beta]
                print('Current best gamma: {} beta: {}'.format(gamma, beta))
    return best_pair, min_rmse_v_arr


def training(sa_train, sa_validation, sa_test, data_path,
             best_params=(0.006, 0.045), k_order=28, continue_from_save=False):
    svd_path = '{}/SVD_path'.format(data_path)
    svd_cluster = SvdCluster(k_order=k_order,
                             gamma=best_params[0],
                             beta=best_params[1],
                             num_of_iters=101,
                             verbose=True,
                             svd_path=svd_path)
    if continue_from_save:
        svd_cluster.load_svd_params()

    svd_cluster.svd_train(sa_train, sa_validation, sa_test, print_step_size=3)
    svd_cluster.save_svd_params()
    svd_cluster.plot_progress(print_step_size=3)
    return svd_cluster


def load_data_and_train(data_path):
    sa_train, sa_validation, sa_test = load_train_validation_test(data_path)

    sa_train.data = sa_train.data.astype(np.double) / 10
    sa_validation.data = sa_validation.data.astype(np.double) / 10
    sa_test.data = sa_test.data.astype(np.double) / 10

    movie_nonzero = sa_train.nonzero()
    movie_nonzero = np.unique(movie_nonzero[1])
    svd_cluster = training(sa_train, sa_validation, sa_test, data_path)
    svd_cluster.remove_zero_rating_movies(movie_nonzero)


if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)

    smartqi = SmartQi(data_path)
    smartqi.give_n_recommendations([122904, 122918], [10, -5])

