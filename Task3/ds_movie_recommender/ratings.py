import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from matplotlib import pyplot as plt


def ratings_to_sparse_matrix(data_path):
    movie_lens_path = '{}/ratings_information'.format(data_path)

    ratings = pd.read_csv('{}/{}.csv'.format(movie_lens_path, 'ratings'))
    ratings.drop_duplicates()

    del ratings['timestamp']

    row_len = 270896
    col_len = 176279

    row = list(ratings.userId.apply(lambda x: x-1))
    col = list(ratings.movieId.apply(lambda x: x-1))

    data = ratings['rating'].tolist()
    sparse_matrix = csr_matrix((data, (row, col)), shape=(row_len, col_len))

    sparse_matrix = sparse_matrix.astype(np.float16)  # we don't need high precision

    save_npz('{}/{}.npz'.format(data_path, 'sparse_rating_matrix'), sparse_matrix)


def remove_rows_with_less_than_n(sparse_matrix, n=10, data_path=None):
    sparse_matrix = sparse_matrix[sparse_matrix.getnnz(1) > n]
    if data_path:
        save_npz('{}/{}.npz'.format(data_path, 'colab_filt_rm'), sparse_matrix)
    return sparse_matrix


def visualize_sparse(sparse_matrix):
    num_of_ratings_per_user = np.sort(np.diff(sparse_matrix.indptr))
    last_num = 0
    for i in range(1, 11):
        print('users with {} ratings'.format(i), end=' ')
        print(np.where(num_of_ratings_per_user > i)[0][0] - last_num)
        last_num = np.where(num_of_ratings_per_user > i)[0][0]
    print(np.where(num_of_ratings_per_user > 10)[0][0])
    print(num_of_ratings_per_user[-10:])

    plt.figure()
    start = np.log10(num_of_ratings_per_user.min())
    stop = np.log10(num_of_ratings_per_user.max())
    bins = np.logspace(start, stop, num=50, endpoint=True)

    plt.xscale('log')
    plt.xticks(bins, np.floor(bins).astype(np.uint16), rotation='vertical')

    plt.hist(num_of_ratings_per_user, bins=bins, log=False)

    plt.show()


def main():
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)

    sparse_matrix = load_npz('{}/{}.npz'.format(data_path, 'sparse_rating_matrix'))

    # visualize_sparse(sparse_matrix)

    coo_matrix = sparse_matrix.tocoo(copy=False)
    return coo_matrix


if __name__ == '__main__':
    ret = main()
