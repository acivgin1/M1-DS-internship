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
    svd_cluster = SvdCluster(k_order=36, gamma=0.005, beta=0.03, num_of_iters=10, verbose=True, svd_path=svd_path)

    # mu = (sA_train.data.sum()/sA_train.data.shape)[0]
    svd_cluster.svd_train(sA_train, sA_validation, sA_test)
    svd_cluster.save_svd_params()

    svd_cluster.plot_progress()

    # joined_df = pd.read_csv('{}/imdb_movielens.csv'.format(data_path), index_col='movieId')
    # joined_df.index -= 1
    # Qi = pd.DataFrame(svd_cluster.qi)
    #
    # s = joined_df.join(Qi)

