cimport cython
import numpy as np
cimport numpy as np
import time

def svd_train(R, V, k_order=100, gamma=0.002, beta=0.01, num_of_iters=20, v_range=(1, 5), print_state=True):
    '''
    performs stochastic gradient descent on SVD algorithm

    :param R: a sparse COOrdinate matrix, list of (i, j, value) of Rating data
    :param V: a sparse COOrdinate matrix, list of (i, j, value) for Validation
    :param k_order: k-th order approximation
    :param gamma: learning rate
    :param beta: regularization parameter
    :param num_of_iters: number of iterations
    :param v_range: tuple specifying values range
    :param print_state: print_state errors
    :return: mu, bu, bi, pu, qi
    '''
    cdef np.ndarray[np.double_t] rmse_arr
    cdef np.ndarray[np.double_t] rmse_t_arr

    cdef np.ndarray[np.double_t] bu
    cdef np.ndarray[np.double_t] bi

    cdef np.ndarray[np.double_t, ndim=2] pu
    cdef np.ndarray[np.double_t, ndim=2] qi

    cdef int u, i, k
    cdef double mean, std_dev, one_minus_gb, mu, r, err, dot, puk, qik

    mean = 0
    std_dev = 0.1
    one_minus_gb = 1 - gamma * beta

    mu = (R.data.sum()/R.data.shape)[0]

    rmse_arr = np.array([])
    rmse_t_arr = np.array([])

    bu = np.zeros(R.shape[0], dtype=np.double)
    bi = np.zeros(R.shape[1], dtype=np.double)

    pu = np.random.normal(mean, std_dev, (R.shape[0], k_order))
    qi = np.random.normal(mean, std_dev, (R.shape[1], k_order))

    for iteration in range(num_of_iters):
        start = time.time()
        cum_error = 0
        for u, i, r in zip(R.row, R.col, R.data):
            dot = 0
            for k in range(k_order):
                dot += pu[u, k] * qi[i, k]
            error = r - (mu + bu[u] + bi[i] + dot)
            cum_error += error * error

            error *= gamma

            bu[u] += gamma*(error - beta * bu[u])
            bi[i] += gamma*(error - beta * bi[i])

            for k in range(k_order):
                puk = pu[u, k]
                qik = qi[i, k]

                pu[u, k] *= one_minus_gb
                pu[u, k] += error*qik

                qi[i, k] *= one_minus_gb
                qi[i, k] += error*puk


        cum_t_error = 0
        for u, i, r in zip(V.row, V.col, V.data):
            r_hat = svd_predict(u, i, pu, qi, mu, bu, bi)
            error = (r - r_hat)**2

            cum_t_error += error

        rmse = np.sqrt(cum_error/R.data.shape[0])
        rmse_t = np.sqrt(cum_t_error/V.data.shape[0])

        np.append(rmse_arr, rmse)
        np.append(rmse_t_arr, rmse_t)

        if print_state:
            stop = time.time()
            duration = stop-start
            print('dur: {} epoch: {} error: {} t_error: {}'.format(duration, iteration, rmse, rmse_t))

    return mu, bu, bi, pu, qi, rmse_arr, rmse_t_arr


def svd_predict(u, i, pu, qi, mu, bu=0, bi=0):
    '''
    Returns predicted rating of user with u-id for a movie with i-id

    :param u integer
    :param i: integer
    :param pu: estimated pu matrix
    :param qi: estimated qi matrix
    :param mu: mean of value matrix
    :param bu: user bias vector
    :param bi: item bias vector
    :return: r_ui predicted rating of user for item i
    '''
    cdef int k
    cdef double dot

    dot = 0
    for k in range(qi.shape[1]):
        dot += pu[u, k] * qi[i, k]

    if isinstance(bu, np.ndarray) and isinstance(bi, np.ndarray):
        return mu + bu[u] + bi[i] + dot
    else:
        return mu + dot


def main():
    pass

print('Hello')
if __name__ == '__main__':
    print('Hello')
