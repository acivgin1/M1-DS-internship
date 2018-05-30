import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse import save_npz, load_npz

from visualizations import ratings_per_row, average_ratings_per_row


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


def remove_rows_with_leq_than_n(sparse_matrix, n=1, data_path=None):
    sparse_matrix = sparse_matrix[sparse_matrix.getnnz(1) > n]
    if data_path:
        save_npz('{}/{}.npz'.format(data_path, 'colab_filt_rm'), sparse_matrix)
    return sparse_matrix


def shuffle_sparse(sparse_matrix):
    help = np.vstack((sparse_matrix.row, sparse_matrix.col, sparse_matrix.data)).transpose()

    np.random.shuffle(help)
    np.random.shuffle(help)
    np.random.shuffle(help)

    spar = coo_matrix((help[:, 2], (help[:, 0].astype(np.uint32), help[:, 1].astype(np.uint32))))
    return spar


def partition_data_from_sparse(sm, ratio):
    sm = shuffle_sparse(sm)

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

    train_matrix = coo_matrix((train_list[2, :], (train_list[0, :].astype(np.uint32), train_list[1, :].astype(np.uint32))))
    del train_list
    test_matrix = coo_matrix((test_list[2, :], (test_list[0, :].astype(np.uint32), test_list[1, :].astype(np.uint32))))
    del test_list

    test_matrix = coo_matrix((sm.data[:test_len], (sm.row[:test_len].astype(np.uint32), sm.col[:test_len].astype(np.uint32))))
    train_matrix = coo_matrix((sm.data[test_len:], (sm.row[test_len:].astype(np.uint32), sm.col[test_len:].astype(np.uint32))))

    return train_matrix, test_matrix


def load_train_validation_test(data_path):
    train_matrix = load_npz('{}/train_ratings.npz'.format(data_path))
    validation_matrix = load_npz('{}/validation_ratings.npz'.format(data_path))
    test_matrix = load_npz('{}/test_ratings.npz'.format(data_path))

    return train_matrix, validation_matrix, test_matrix


def main():
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)
    # ratings_to_sparse_matrix(data_path)

    sparse_matrix = load_npz('{}/{}.npz'.format(data_path, 'sparse_rating_matrix'))
    # sparse_matrix = remove_rows_with_leq_than_n(sparse_matrix)
    cA = sparse_matrix.tocoo(copy=False)

    # data1, test = partition_data_from_sparse(cA, ratio=0.8)
    # save_npz('{}/test_ratings.npz'.format(data_path), test)
    #
    # train, validation = partition_data_from_sparse(data1, ratio=0.8)
    # save_npz('{}/train_ratings.npz'.format(data_path), train)
    # save_npz('{}/validation_ratings.npz'.format(data_path), validation)

    train, validation, test = load_train_validation_test(data_path)

    print(train.row.min())
    print(train.row.max())
    print(train.col.min())
    print(train.col.max())
    print()

    print(validation.row.min())
    print(validation.row.max())
    print(validation.col.min())
    print(validation.col.max())
    print()

    print(test.row.min())
    print(test.row.max())
    print(test.col.min())
    print(test.col.max())
    print()

    print(cA.row.min())
    print(cA.row.max())
    print(cA.col.min())
    print(cA.col.max())
    return train, test, validation


if __name__ == '__main__':
    a, b, c = main()
    user_a = a.tocsr()
    user_b = b.tocsr()
    user_c = c.tocsr()

    user_a.data = user_a.data / 10
    user_b.data = user_b.data / 10
    user_c.data = user_c.data / 10

    movie_a = csr_matrix((a.data/10, (a.col, a.row)), shape=(a.col.max() + 1, a.row.max() + 1))
    movie_b = csr_matrix((b.data/10, (b.col, b.row)), shape=(b.col.max() + 1, b.row.max() + 1))
    movie_c = csr_matrix((c.data/10, (c.col, c.row)), shape=(c.col.max() + 1, c.row.max() + 1))

    ratings_per_row(user_a, 'Users', 'Training data: Number of ratings per user', True)
    ratings_per_row(user_b, 'Users', 'Test data: Number of ratings per user', True)
    ratings_per_row(user_c, 'Users', 'Validation: Number of ratings per user', True)

    ratings_per_row(movie_a, 'Movies', 'Training data: Number of ratings per movies', True)
    ratings_per_row(movie_b, 'Movies', 'Test data: Number of ratings per movies', True)
    ratings_per_row(movie_c, 'Movies', 'Validation: Number of ratings per movies', True)

    average_ratings_per_row(user_a, 'Training data: Average rating per user', ymax=120000)
    average_ratings_per_row(user_b, 'Test data: Average rating per user', ymax=90000)
    average_ratings_per_row(user_c, 'Validation: Average rating per user', ymax=80000)

    average_ratings_per_row(movie_a, 'Training data: Average rating per movie')
    average_ratings_per_row(movie_b, 'Test data: Average rating per movie', ymax=10000)
    average_ratings_per_row(movie_c, 'Validation: Average rating per movie', ymax=10000)
