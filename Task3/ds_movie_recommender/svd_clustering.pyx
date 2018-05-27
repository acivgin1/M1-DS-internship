cimport cython
import numpy as np
cimport numpy as np
import time
from matplotlib import pyplot as plt

class SvdCluster:
    def __init__(self, k_order=100, gamma=0.002, beta=0.01, num_of_iters=20, verbose=True, svd_path=None):
        self.k_order = k_order
        self.gamma = gamma
        self.beta = beta
        self.num_of_iters = num_of_iters
        self.verbose = verbose
        self.svd_path = svd_path

        self.mu = np.array([])
        self.bu = np.array([])
        self.bi = np.array([])
        self.pu = np.array([]).reshape(-1, 1)
        self.qi = np.array([]).reshape(-1, 1)

        self.rmse_arr = np.array([])
        self.mae_v_arr = np.array([])
        self.rmse_v_arr = np.array([])
        self.mae_t_arr = np.array([])
        self.rmse_t_arr = np.array([])

        self.pu_dev = np.array([])
        self.qi_dev = np.array([])

    def svd_train(self, R, V, T, modulo=1):
        mu, bu, bi, pu, qi, rmse, ortho_dev = _svd_train(R, V, T,
                                                         self.k_order, self.gamma, self.beta,
                                                         self.num_of_iters, self.verbose, self.svd_path,
                                                         self.bu, self.bi, self.pu, self.qi, modulo)

        self.mu, self.bu, self.bi, self.pu, self.qi = (mu, bu, bi, pu, qi)

        self.rmse_arr = np.append(self.rmse_arr, rmse[0, :])
        self.mae_v_arr = np.append(self.mae_v_arr, rmse[1, :])
        self.rmse_v_arr = np.append(self.rmse_v_arr, rmse[2, :])
        self.mae_t_arr = np.append(self.mae_t_arr, rmse[3, :])
        self.rmse_t_arr = np.append(self.rmse_t_arr, rmse[4, :])

        self.pu_dev = np.append(self.pu_dev, ortho_dev[0, :])
        self.qi_dev = np.append(self.qi_dev, ortho_dev[1, :])

    def svd_predict_dataset(self, T):
        return _svd_predict_dataset(T, self.mu, self.bu, self.bi, self.pu, self.qi)

    def svd_predict(self, u, i):
        return _svd_predict(u, i, self.mu, self.bu, self.bi, self.pu, self.qi)

    def plot_progress(self):
        plt.figure()
        plt.plot(self.rmse_arr)
        plt.plot(self.rmse_v_arr + 0.01)
        plt.plot(self.mae_v_arr + 0.01)
        plt.plot(self.rmse_t_arr)
        plt.plot(self.mae_t_arr)
        plt.grid()
        plt.legend(['Training', 'Validation', 'MAE: validation', 'Testing', 'MAE: testing'])
        plt.xlabel('Iteration')
        plt.ylabel('Errors')
        plt.show()

        plt.figure()
        plt.plot(np.diff(self.rmse_arr))
        plt.plot(np.diff(self.rmse_v_arr))
        plt.plot(np.diff(self.mae_v_arr))
        plt.plot(np.diff(self.rmse_t_arr))
        plt.plot(np.diff(self.mae_t_arr))
        plt.grid()
        plt.legend(['Training', 'Validation', 'MAE: validation', 'Testing', 'MAE: testing'])
        plt.xlabel('Iteration')
        plt.ylabel('Errors first deriv.')
        plt.show()

        plt.figure()
        plt.plot(self.pu_dev)
        plt.plot(self.qi_dev)
        plt.grid()
        plt.legend(['Pu', 'Qi'])
        plt.xlabel('Iteration')
        plt.ylabel('Dev from ortho')
        plt.show()

    def save_svd_params(self):
        np.savez('{}/svd_params.npz'.format(self.svd_path),
                 bu=self.bu, bi=self.bi, pu=self.pu, qi=self.qi,
                 pu_dev=self.pu_dev, qi_dev=self.qi_dev,
                 rmse_arr=self.rmse_arr, rmse_v_arr=self.rmse_v_arr, mae_v_arr=self.mae_v_arr,
                 rmse_t_arr=self.rmse_t_arr, mae_t_arr=self.mae_t_arr)

    def load_svd_params(self, mu):
        self.mu = mu
        loadz = np.load('{}/svd_params.npz'.format(self.svd_path))

        self.bu = loadz['bu']
        self.bi = loadz['bi']
        self.pu = loadz['pu']
        self.qi = loadz['qi']

        self.pu_dev = loadz['pu_dev']
        self.qi_dev = loadz['qi_dev']

        self.rmse_arr = loadz['rmse_arr']
        self.rmse_v_arr = loadz['rmse_v_arr']
        self.mae_v_arr = loadz['mae_v_arr']
        self.rmse_t_arr = loadz['rmse_t_arr']
        self.mae_t_arr = loadz['mae_t_arr']


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _svd_train(R, V, T, int k_order, double gamma, double beta, int num_of_iters, verbose, svd_path,
               np.ndarray[np.double_t] bu_in, np.ndarray[np.double_t] bi_in,
               np.ndarray[np.double_t, ndim=2] pu_in, np.ndarray[np.double_t, ndim=2] qi_in, modulo=1):
    cdef np.ndarray[np.double_t, ndim=2] rmse_arr, ortho_dev, pu, qi
    cdef np.ndarray[np.double_t] bu, bi

    cdef int u, i, k
    cdef double mean, std_dev, one_minus_gb, mu
    cdef double r, r_hat, error, cum_error, cum_t_error, cum_v_error
    cdef double dot, puk, qik
    cdef double rmse, rmse_t, mae_t, rmse_v, mae_v

    mu = (R.data.sum()/R.data.shape)[0]  # andrew NG recommends this

    mean = np.sqrt(mu/k_order)
    std_dev = 0.1
    one_minus_gb = 1 - gamma * beta

    rmse_arr = np.empty((5, round(num_of_iters/modulo + .5)), dtype=np.double)
    ortho_dev = np.empty((2, round(num_of_iters/modulo + .5)), dtype=np.double)

    if bu_in.size == 0:
        bu = np.random.normal(0, std_dev/10, R.shape[0])
        bi = np.random.normal(0, std_dev/10, R.shape[1])
        pu = np.random.normal(mean, std_dev, (R.shape[0], k_order))
        qi = np.random.normal(mean, std_dev, (R.shape[1], k_order))
    else:
        bu = bu_in
        bi = bi_in
        pu = pu_in
        qi = qi_in

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
            bu[u] *= one_minus_gb
            bi[i] *= one_minus_gb

            bu[u] += error
            bi[i] += error

            for k in range(k_order):
                puk = pu[u, k]
                qik = qi[i, k]

                pu[u, k] *= one_minus_gb
                pu[u, k] += error*qik

                qi[i, k] *= one_minus_gb
                qi[i, k] += error*puk


        rmse = np.sqrt(cum_error/R.data.shape[0])

        if iteration % modulo == 0:
            # Fadile oprosti
            mae_v = 0
            cum_v_error = 0
            for u, i, r in zip(V.row, V.col, V.data):
                dot = 0
                for k in range(qi.shape[1]):
                    dot += pu[u, k] * qi[i, k]
                r_hat = mu + bu[u] + bi[i] + dot
                error = r - r_hat
                mae_v += abs(error)

                error *= error
                cum_v_error += error
            mae_v = mae_v/V.data.shape[0]
            rmse_v = np.sqrt(cum_v_error/V.data.shape[0])

            mae_t = 0
            cum_t_error = 0
            for u, i, r in zip(T.row, T.col, T.data):
                dot = 0
                for k in range(qi.shape[1]):
                    dot += pu[u, k] * qi[i, k]
                r_hat = mu + bu[u] + bi[i] + dot
                error = r - r_hat
                mae_t += abs(error)

                error *= error
                cum_t_error += error
            mae_t = mae_t/T.data.shape[0]
            rmse_t = np.sqrt(cum_t_error/T.data.shape[0])

            rmse_arr[0, iteration // modulo] = rmse
            rmse_arr[1, iteration // modulo] = mae_v
            rmse_arr[2, iteration // modulo] = rmse_v
            rmse_arr[3, iteration // modulo] = mae_t
            rmse_arr[4, iteration // modulo] = rmse_t

            ortho_dev[0, iteration // modulo] = deviation_from_ortho(pu)
            ortho_dev[1, iteration // modulo] = deviation_from_ortho(qi)

            np.savez('{}/temp_{}.npz'.format(svd_path, iteration), bu=bu, bi=bi, pu=pu, qi=qi)

            if verbose:
                stop = time.time()
                duration = stop-start
                _results = 'rmse_t:{:.6f} mae_t:{:.6f} P_dev:{:.6f}, Q_dev:{:.6f}'.format(rmse_t, mae_t,
                                                                                          ortho_dev[0, iteration//modulo],
                                                                                          ortho_dev[1, iteration//modulo])
                print('t:{:.2f} it:{} rmse:{:.6f} rmse_v:{:.6f} mae_v:{:.6f} {}'.format(duration,
                                                                                        iteration,
                                                                                        rmse,
                                                                                        rmse_v,
                                                                                        mae_v,
                                                                                        _results))

        else:
            stop = time.time()
            duration = stop-start
            print('t:{:.2f}'.format(duration))
    return mu, bu, bi, pu, qi, rmse_arr, ortho_dev


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _svd_predict_dataset(T, mu,
                          np.ndarray[np.double_t] bu, np.ndarray[np.double_t] bi,
                          np.ndarray[np.double_t, ndim=2] pu, np.ndarray[np.double_t, ndim=2] qi):
    cdef int k
    cdef double mae_t, cum_t_error, r_hat, error, dot
    mae_t = 0
    cum_t_error = 0
    for u, i, r in zip(T.row, T.col, T.data):
        dot = 0
        for k in range(qi.shape[1]):
            dot += pu[u, k] * qi[i, k]
        r_hat = mu + bu[u] + bi[i] + dot
        error = r - r_hat
        mae_t += abs(error)

        error *= error
        cum_t_error += error
    mae_t = mae_t/T.data.shape[0]
    rmse_t = np.sqrt(cum_t_error/T.data.shape[0])
    return mae_t, rmse_t


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def _svd_predict(u, i, mu,
                 np.ndarray[np.double_t] bu, np.ndarray[np.double_t] bi,
                 np.ndarray[np.double_t, ndim=2] pu, np.ndarray[np.double_t, ndim=2] qi):
    cdef int k
    cdef double dot

    dot = 0
    for k in range(qi.shape[1]):
        dot += pu[u, k] * qi[i, k]
    return mu + bu[u] + bi[i] + dot

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def deviation_from_ortho(np.ndarray[np.double_t, ndim=2] M, size=1000):
    cdef np.ndarray[np.double_t, ndim=2] mask
    cdef np.ndarray[np.double_t, ndim=2] extract
    mask = np.ones((size, size)) - np.eye(size)
    extract = M[0:size, :]
    return np.abs(mask*(np.dot(extract, extract.transpose()))).sum()/size/size


print('svd_clustering.pyx - build successful')
if __name__ == '__main__':
    print('Hello')
