import os

import numpy as np
from ratings import load_train_test
from matplotlib import pyplot as plt
import pyximport; pyximport.install()
from matrix_factorization import SvdCluster


def main():
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    svd_path = '{}/SVD_path'.format(data_path)

    sA_train, sA_test = load_train_test(data_path)

    svd_cluster = SvdCluster(k_order=100, gamma=0.005, beta=0.02, num_of_iters=2, verbose=True)

    mu = (sA_train.data.sum()/sA_train.data.shape)[0]
    svd_cluster.load_svd_params(svd_path, mu)
    svd_cluster.svd_train(sA_train, sA_test)
    # svd_cluster.save_svd_params(svd_path)

    svd_cluster.plot_progress()


if __name__ == '__main__':
    main()
