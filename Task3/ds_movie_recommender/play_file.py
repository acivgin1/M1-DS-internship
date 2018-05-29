import os

import numpy as np
import pandas as pd
from ratings import load_train_validation_test
import pyximport; pyximport.install()
from svd_clustering import SvdCluster


if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    svd_path = '{}/SVD_path'.format(data_path)

    sA_train, sA_validation, sA_test = load_train_validation_test(data_path)
    sA_train.data = sA_train.data.astype(np.double) / 10
    sA_validation.data = sA_validation.data.astype(np.double) / 10
    sA_test.data = sA_test.data.astype(np.double) / 10

    # num of params (k_order + 1)*(270895 + 176274), for k_order=36 this is 16545253, and we have 16655546 train ratings

    # gamma_list = [0.006, 0.005]
    # beta_list = [0.06, 0.05, 0.04]
    # min_rmse_v_arr = 100
    # best_pair = []

    # for gamma in gamma_list:
    #     for beta in beta_list:
    #         svd_cluster = SvdCluster(k_order=28, gamma=gamma, beta=beta,
    #                                  num_of_iters=61, verbose=True, svd_path=svd_path)
    #         svd_cluster.svd_train(sA_train, sA_validation, sA_test, print_step_size=30)
    #         if np.where(np.diff(svd_cluster.rmse_v_arr) > 0)[0].size > 0:
    #             print('Error is rising')
    #             continue
    #         if svd_cluster.rmse_v_arr[-1] < min_rmse_v_arr:
    #             min_rmse_v_arr = svd_cluster.rmse_v_arr[-1]
    #             best_pair = [gamma, beta]
    #             print('Current best gamma: {} beta: {}'.format(gamma, beta))
    svd_cluster = SvdCluster(k_order=28, gamma=0.006, beta=0.045, num_of_iters=101, verbose=True, svd_path=svd_path)

    mu = sA_train.data.sum()/sA_train.data.size
    svd_cluster.load_svd_params(mu)

    # svd_cluster.svd_train(sA_train, sA_validation, sA_test, print_step_size=5)
    # svd_cluster.save_svd_params()
    svd_cluster.plot_progress(print_step_size=5)
