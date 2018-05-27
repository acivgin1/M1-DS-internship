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
    sA_train.data /= 10
    sA_validation.data /= 10
    sA_test.data /= 10

    # num of params (k_order + 1)*(270895 + 176274), for k_order=36 this is 16545253, and we have 16655546 train ratings
    gamas = [0.001]
    betas = [0.01, 0.03, 0.001]

    max_rmse_v_fall = 0
    best_pair = [0, 0]
    for gama in gamas:
        for beta in betas:
            svd_cluster = SvdCluster(k_order=36, gamma=gama, beta=beta,
                                     num_of_iters=51, verbose=True, svd_path=svd_path)
            svd_cluster.svd_train(sA_train, sA_validation, sA_test, modulo=10)
            print(svd_cluster.rmse_v_arr)
            if np.where(np.diff(svd_cluster.rmse_v_arr) > 0)[0].size > 0:
                print('Error is rising')
                continue
            if -np.diff(svd_cluster.rmse_v_arr)[-1] > max_rmse_v_fall:
                max_rmse_v_fall = -np.diff(svd_cluster.rmse_v_arr)[-1]
                best_pair = [gama, beta]
    # svd_cluster = SvdCluster(k_order=36, gamma=0.005, beta=0.03, num_of_iters=10, verbose=True, svd_path=svd_path)
    print(best_pair)
    # mu = (sA_train.data.sum()/sA_train.data.shape)[0]
    # svd_cluster.svd_train(sA_train, sA_validation, sA_test)
    # svd_cluster.save_svd_params()

    # svd_cluster.plot_progress()

    # joined_df = pd.read_csv('{}/imdb_movielens.csv'.format(data_path), index_col='movieId')
    # joined_df.index -= 1
    # Qi = pd.DataFrame(svd_cluster.qi)
    #
    # s = joined_df.join(Qi)

