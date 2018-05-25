import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import save_npz, load_npz


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

    sparse_matrix.data *= 10
    sparse_matrix = sparse_matrix.astype(np.uint32)  # we don't need high precision

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


def shuffle_sparse(sparse_matrix):
    help = np.vstack((sparse_matrix.row, sparse_matrix.col, sparse_matrix.data)).transpose()

    np.random.shuffle(help)

    spar = coo_matrix((help[:, 2], (help[:, 0].astype(np.uint32), help[:, 1].astype(np.uint32))))
    return spar


def train_and_test_from_sparse(sm, data_path, ratio):
    test_len = int(sm.data.shape[0] * (1 - ratio))

    test_list = np.empty([3, test_len])
    train_list = np.empty([3, sm.data.shape[0] - test_len])

    test_i = 0
    train_i = 0

    seen_row_set = set()
    seen_col_set = set()

    for i in tqdm(range(sm.data.shape[0])):
        if test_i < test_len and sm.row[i] in seen_row_set and sm.col[i] in seen_col_set:
            test_list[:, test_i] = np.array([sm.row[i], sm.col[i], sm.data[i]])
            test_i += 1
        else:
            train_list[:, train_i] = np.array([sm.row[i], sm.col[i], sm.data[i]])
            train_i += 1
            seen_row_set.add(sm.row[i])
            seen_col_set.add(sm.col[i])

    # train_matrix = sparse.coo_matrix((sm.data[0:train_len], (sm.row[0:train_len], sm.col[0:train_len])))
    # test_matrix = sparse.coo_matrix((sm.data[train_len:], (sm.row[train_len:], sm.col[train_len:])))
    train_matrix = coo_matrix((train_list[2, :], (train_list[0, :].astype(np.uint32), train_list[1, :].astype(np.uint32))))
    del train_list
    test_matrix = coo_matrix((test_list[2, :], (test_list[0, :].astype(np.uint32), test_list[1, :].astype(np.uint32))))
    del test_list

    save_npz('{}/train_ratings.npz'.format(data_path), train_matrix)
    save_npz('{}/test_ratings.npz'.format(data_path), test_matrix)


def load_train_test(data_path):
    train_matrix = load_npz('{}/train_ratings.npz'.format(data_path))
    test_matrix = load_npz('{}/test_ratings.npz'.format(data_path))
    return train_matrix, test_matrix


def main():
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)

    # ratings_to_sparse_matrix(data_path)

    sparse_matrix = load_npz('{}/{}.npz'.format(data_path, 'sparse_rating_matrix'))

    # visualize_sparse(sparse_matrix)

    cA = sparse_matrix.tocoo(copy=False)

    # cA = shuffle_sparse(cA)
    # train_and_test_from_sparse(cA, data_path, ratio=0.8)
    train, test = load_train_test(data_path)

    print(train.row.min())
    print(train.row.max())
    print(train.col.min())
    print(train.col.max())

    print(test.row.min())
    print(test.row.max())
    print(test.col.min())
    print(test.col.max())

    print(cA.row.min())
    print(cA.row.max())
    print(cA.col.min())
    print(cA.col.max())
    return cA


if __name__ == '__main__':
    ret = main()
