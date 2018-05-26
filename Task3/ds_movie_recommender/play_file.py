import os

import numpy as np
import pandas as pd
from ratings import load_train_test
from matplotlib import pyplot as plt
import pyximport; pyximport.install()
from matrix_factorization import SvdCluster


if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    svd_path = '{}/SVD_path'.format(data_path)

    sA_train, sA_test = load_train_test(data_path)
    sA_train.data /= 10
    sA_test.data /= 10

    svd_cluster = SvdCluster(k_order=75, gamma=0.005, beta=0.02, num_of_iters=100, verbose=True)

    mu = (sA_train.data.sum()/sA_train.data.shape)[0]
    svd_cluster.svd_train(sA_train, sA_test)
    svd_cluster.save_svd_params(svd_path)

    svd_cluster.plot_progress()

    joined_df = pd.read_csv('{}/imdb_movielens.csv'.format(data_path), index_col='movieId')
    joined_df.index -= 1
    Qi = pd.DataFrame(svd_cluster.qi)


