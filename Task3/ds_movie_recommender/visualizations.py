import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz


def ratings_per_row(sparse_matrix, element_name, title, log_scale=False):
    num_of_ratings_per_row = np.sort(np.diff(sparse_matrix.indptr))

    last_num = 0
    for i in range(0, 11):
        print('{} with {} ratings'.format(element_name, i), end=' ')
        print(np.where(num_of_ratings_per_row > i)[0][0] - last_num)
        last_num = np.where(num_of_ratings_per_row > i)[0][0]
    print(np.where(num_of_ratings_per_row > 10)[0][0])
    print(num_of_ratings_per_row[-10:])

    reduce_by_one = False
    if num_of_ratings_per_row.min() == 0:
        num_of_ratings_per_row += 1
        reduce_by_one = True

    plt.figure()
    start = np.log10(num_of_ratings_per_row.min())
    stop = np.log10(num_of_ratings_per_row.max())
    bins = np.logspace(start, stop, num=50, endpoint=True)

    plt.xscale('log')
    if reduce_by_one:
        plt.xticks(bins, np.floor(bins).astype(np.uint16)-1, rotation='vertical')
    else:
        plt.xticks(bins, np.floor(bins).astype(np.uint16), rotation='vertical')

    plt.xlabel('Num of ratings')
    plt.ylabel('Size of bin')
    plt.title(title)

    plt.hist(num_of_ratings_per_row, bins=bins, log=log_scale)

    plt.show()


def average_ratings_per_row(sparse_matrix, title, ymax=16000):
    num_of_ratings_per_row = np.diff(sparse_matrix.indptr).reshape((-1, 1))
    sum_of_ratings_per_row = sparse_matrix.sum(axis=1)

    plt.xlabel('Average rating')
    plt.ylabel('Number of rows')
    plt.title(title)

    np.seterr(divide='ignore')
    res = np.nan_to_num(np.divide(sum_of_ratings_per_row, num_of_ratings_per_row))
    # hist = res
    # if res.min() < 0.1:
    #     hist = res[np.where(res > 0)]

    plt.hist(res, bins=10)
    plt.xlim(xmin=0.5)
    if res.min() < 0.1:
        plt.ylim(ymax=ymax)

    plt.show()


if __name__ == '__main__':
    cur_path = os.path.dirname(__file__)
    data_path = os.path.relpath('../Data', cur_path)

    user_matrix = load_npz('{}/{}.npz'.format(data_path, 'sparse_rating_matrix'))
    user_matrix.data = user_matrix.data / 10

    coo = user_matrix.tocoo()
    movie_matrix = csr_matrix((coo.data, (coo.col, coo.row)), shape=(coo.col.max()+1, coo.row.max()+1))
    del coo

    ratings_per_row(user_matrix, 'Users', 'Number of ratings per user', log_scale=False)
    ratings_per_row(movie_matrix, 'Movies', 'Number of ratings per film', log_scale=True)

    average_ratings_per_row(user_matrix, 'Average ratings per user')
    average_ratings_per_row(movie_matrix, 'Average ratings per movie')
