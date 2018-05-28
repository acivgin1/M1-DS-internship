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

    gamma_list = [0.03, 0.01, 0.003, 0.001]
    beta_list = [0.3, 0.1, 0.03, 0.01, 0.003]
    min_rmse_v_arr = 100

    for gamma in gamma_list:
        for beta in beta_list:
            svd_cluster = SvdCluster(k_order=28, gamma=gamma, beta=beta,
                                     num_of_iters=51, verbose=True, svd_path=svd_path)
            svd_cluster.svd_train(sA_train, sA_validation, sA_test, print_step_size=25)
            if np.where(np.diff(svd_cluster.rmse_v_arr) > 0)[0].size > 0:
                print('Error is rising')
                continue
            if svd_cluster.rmse_v_arr[-1] < min_rmse_v_arr:
                min_rmse_v_arr = svd_cluster.rmse_v_arr[-1]
                best_pair = [gamma, beta]
                print('Current best gamma: {} beta: {}'.format(gamma, beta))
    # svd_cluster = SvdCluster(k_order=75, gamma=0.005, beta=0.02, num_of_iters=100, verbose=True, svd_path=svd_path)
    # print(best_k_order)
    # mu = (sA_train.data.sum()/sA_train.data.shape)[0]
    # svd_cluster.svd_train(sA_train, sA_validation, sA_test, print_step_size=10)
    # svd_cluster.save_svd_params()

    # svd_cluster.plot_progress()

    # joined_df = pd.read_csv('{}/imdb_movielens.csv'.format(data_path), index_col='movieId')
    # joined_df.index -= 1
    # Qi = pd.DataFrame(svd_cluster.qi)
    # s = joined_df.join(Qi)
    #



