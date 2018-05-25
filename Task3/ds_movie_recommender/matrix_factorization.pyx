cimport cython
import numpy as np
cimport numpy as np
import time


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def svd_train(R, V, int k_order=100, double gamma=0.002, double beta=0.01, int num_of_iters=20, print_state=True):
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
    cdef np.ndarray[np.double_t] mae_t_arr

    cdef np.ndarray[np.double_t] bu
    cdef np.ndarray[np.double_t] bi

    cdef np.ndarray[np.double_t, ndim=2] pu
    cdef np.ndarray[np.double_t, ndim=2] qi

    cdef int u, i, k
    cdef double mean, std_dev, one_minus_gb, mu
    cdef double r, r_hat, error, cum_error, cum_t_error
    cdef double dot, puk, qik
    cdef double rmse, rmse_t, mae_t

    mu = (R.data.sum()/R.data.shape)[0]  # andrew NG recommends this

    mean = np.sqrt(mu/k_order)
    std_dev = 0.1
    one_minus_gb = 1 - gamma * beta

    rmse_arr = np.empty(num_of_iters, dtype=np.double)
    rmse_t_arr = np.empty(num_of_iters, dtype=np.double)
    mae_t_arr = np.empty(num_of_iters, dtype=np.double)

    # bu = np.zeros(R.shape[0], dtype=np.double)
    # bi = np.zeros(R.shape[1], dtype=np.double)

    bu = np.random.normal(0, std_dev/10, R.shape[0])
    bi = np.random.normal(0, std_dev/10, R.shape[1])

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
        mae_t = 0
        for u, i, r in zip(V.row, V.col, V.data):
            r_hat = svd_predict(u, i, pu, qi, mu, bu, bi)
            error = r - r_hat
            mae_t += abs(error)

            error *= error
            cum_t_error += error

        rmse = np.sqrt(cum_error/R.data.shape[0])
        rmse_t = np.sqrt(cum_t_error/V.data.shape[0])
        mae_t = mae_t/V.data.shape[0]

        rmse_arr[iteration] = rmse
        rmse_t_arr[iteration] = rmse_t
        mae_t_arr[iteration] = mae_t

        if print_state:
            stop = time.time()
            duration = stop-start
            print('t:{:.2f} it:{} rmse:{:.6f} rmse_t:{:.6f} mae_t:{:.6f}'.format(duration, iteration, rmse, rmse_t, mae_t))

    return mu, bu, bi, pu, qi, rmse_arr, rmse_t_arr, mae_t_arr


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def svd_predict(u, i, np.ndarray[np.double_t, ndim=2] pu, np.ndarray[np.double_t, ndim=2] qi, mu, np.ndarray[np.double_t] bu, np.ndarray[np.double_t] bi):
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

    return mu + bu[u] + bi[i] + dot


def svd_test(test_set, mu, bu, bi, pu, qi):
    cum_error = 0
    for u, i, r in zip(test_set.row, test_set.col, test_set.data):
        r_hat = svd_predict(u, i, pu, qi, mu, bu, bi)
        error = (r - r_hat)**2

        cum_error += error

    rmse = np.sqrt(cum_error / test_set.data.shape[0])
    return rmse


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def deviation_from_ortho(np.ndarray[np.double_t, ndim=2] M, size=100):
    shape = M.shape
    return np.abs((np.ones((size, size))-np.eye(size, size))*M[0:size, 0:size]).sum()/size/size

def main():
    pass

print('Hello')
if __name__ == '__main__':
    print('Hello')
