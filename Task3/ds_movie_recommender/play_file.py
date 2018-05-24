import numpy as np
from scipy import sparse
from ratings import main as r_main
from matplotlib import pyplot as plt
from assymetric_svd import svd_predict, svd_train

# import pyximport; pyximport.install()
# from matrix_factorization import svd_predict, svd_train


def shuffle_sparse(sparse_matrix):
    rng_state = np.random.get_state()
    np.random.shuffle(sparse_matrix.data)
    np.random.set_state(rng_state)
    np.random.shuffle(sparse_matrix.row)
    np.random.set_state(rng_state)
    np.random.shuffle(sparse_matrix.col)


def train_and_test_from_sparse(sm, ratio=0.7):
    train_len = int(sm.data.shape[0] * ratio)
    train_matrix = sparse.coo_matrix((sm.data[0:train_len], (sm.row[0:train_len], sm.col[0:train_len])))
    test_matrix = sparse.coo_matrix((sm.data[train_len:], (sm.row[train_len:], sm.col[train_len:])))
    return train_matrix, test_matrix


A = r_main()

sA = sparse.coo_matrix(A)
shuffle_sparse(sA)

del A

# sA, _ = train_and_test_from_sparse(sA, ratio=1)

# sA = sA.tocsr()
# sA = sA[sA.getnnz(1) > 0][:, sA.getnnz(0) > 0]
# sA = sA.tocoo()

sA_train, sA_test = train_and_test_from_sparse(sA, ratio=0.8)
del sA

mu, bu, bi, pu, qi, rmse_arr, rmse_t_arr = svd_train(sA_train, sA_test,
                                                     k_order=20,
                                                     gamma=0.005,
                                                     beta=0.02,
                                                     num_of_iters=20,
                                                     print_state=True)

n = len(rmse_arr)
t = np.arange(1, n+1)

rmse_arr = np.array(rmse_arr)
rmse_t_arr = np.array(rmse_t_arr)

plt.figure()
plt.subplot(211)
plt.plot(t, rmse_arr)
plt.plot(t, rmse_t_arr)
plt.legend(['Training', 'Testing'])
plt.xlabel('Broj iteracija')
plt.ylabel('RMSE')

plt.subplot(212)
plt.plot(t[1:], -np.diff(rmse_arr))
plt.plot(t[1:], -np.diff(rmse_t_arr))
plt.legend(['Training', 'Testing'])
plt.xlabel('Broj iteracija')
plt.ylabel('RMSE')
