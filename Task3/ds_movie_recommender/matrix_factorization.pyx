cimport cython
import numpy as np
cimport numpy as np
import time
from matplotlib import pyplot as plt

class SvdCluster:
    def __init__(self, k_order=100, gamma=0.002, beta=0.01, num_of_iters=20, verbose=True):
        self.k_order = k_order
        self.gamma = gamma
        self.beta = beta
        self.num_of_iters = num_of_iters
        self.verbose = verbose

        self.mu = None
        self.bu = None
        self.bi = None
        self.pu = None
        self.qi = None

        self.rmse_arr = np.array([])
        self.rmse_t_arr = np.array([])
        self.mae_t_arr = np.array([])

        self.pu_dev = np.array([])
        self.qi_dev = np.array([])

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def svd_train(self, R, V):
        cdef np.ndarray[np.double_t] rmse_arr
        cdef np.ndarray[np.double_t] rmse_t_arr
        cdef np.ndarray[np.double_t] mae_t_arr

        cdef np.ndarray[np.double_t] pu_dev
        cdef np.ndarray[np.double_t] qi_dev

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

        mean = np.sqrt(mu/self.k_order)
        std_dev = 0.1
        one_minus_gb = 1 - self.gamma * self.beta

        rmse_arr = np.empty(self.num_of_iters, dtype=np.double)
        rmse_t_arr = np.empty(self.num_of_iters, dtype=np.double)
        mae_t_arr = np.empty(self.num_of_iters, dtype=np.double)

        pu_dev = np.empty(self.num_of_iters, dtype=np.double)
        qi_dev = np.empty(self.num_of_iters, dtype=np.double)

        # bu = np.zeros(R.shape[0], dtype=np.double)
        # bi = np.zeros(R.shape[1], dtype=np.double)

        bu = np.random.normal(0, std_dev/10, R.shape[0])
        bi = np.random.normal(0, std_dev/10, R.shape[1])

        pu = np.random.normal(mean, std_dev, (R.shape[0], self.k_order))
        qi = np.random.normal(mean, std_dev, (R.shape[1], self.k_order))

        for iteration in range(self.num_of_iters):
            start = time.time()
            cum_error = 0
            for u, i, r in zip(R.row, R.col, R.data):
                dot = 0
                for k in range(self.k_order):
                    dot += pu[u, k] * qi[i, k]
                error = r - (mu + bu[u] + bi[i] + dot)
                cum_error += error * error

                error *= self.gamma

                bu[u] += self.gamma*(error - self.beta * bu[u])
                bi[i] += self.gamma*(error - self.beta * bi[i])

                for k in range(self.k_order):
                    puk = pu[u, k]
                    qik = qi[i, k]

                    pu[u, k] *= one_minus_gb
                    pu[u, k] += error*qik

                    qi[i, k] *= one_minus_gb
                    qi[i, k] += error*puk

            self.mu = mu
            self.bu = bu
            self.bi = bi
            self.pu = pu
            self.qi = qi

            cum_t_error = 0
            mae_t = 0
            for u, i, r in zip(V.row, V.col, V.data):
                r_hat = self.svd_predict(u, i)
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

            pu_dev[iteration] = deviation_from_ortho(pu)
            qi_dev[iteration] = deviation_from_ortho(qi)
            if self.verbose:
                stop = time.time()
                duration = stop-start
                print('t:{:.2f} it:{} rmse:{:.6f} rmse_t:{:.6f} mae_t:{:.6f}'.format(duration, iteration, rmse, rmse_t, mae_t))
                print('Pu_ort_dev:{:.6f}, Qi_ort_dev:{:.6f}'.format(pu_dev[iteration], qi_dev[iteration]))

        self.rmse_arr = np.append(self.rmse_arr, rmse_arr)
        self.rmse_t_arr = np.append(self.rmse_t_arr, rmse_t_arr)
        self.mae_t_arr = np.append(self.mae_t_arr, mae_t_arr)

        self.pu_dev = np.append(self.pu_dev, pu_dev)
        self.qi_dev = np.append(self.qi_dev, qi_dev)

    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def svd_predict(self, u, i):
        cdef int k
        cdef double dot

        dot = 0
        for k in range(self.qi.shape[1]):
            dot += self.pu[u, k] * self.qi[i, k]

        return self.mu + self.bu[u] + self.bi[i] + dot

    def svd_test(self, test_set):
        cum_error = 0
        for u, i, r in zip(test_set.row, test_set.col, test_set.data):
            r_hat = self.svd_predict(u, i)
            error = (r - r_hat)**2
            cum_error += error

        rmse = np.sqrt(cum_error / test_set.data.shape[0])
        return rmse

    def plot_progress(self):
        plt.figure()
        plt.plot(self.rmse_arr)
        plt.plot(self.rmse_t_arr)
        plt.plot(self.mae_t_arr)
        plt.grid()
        plt.legend(['Training', 'Testing', 'MAE: testing'])
        plt.xlabel('Iteration')
        plt.ylabel('Errors')
        plt.show()

        plt.figure()
        plt.plot(-np.diff(self.rmse_arr))
        plt.plot(-np.diff(self.rmse_t_arr))
        plt.plot(-np.diff(self.mae_t_arr))
        plt.grid()
        plt.legend(['Training', 'Testing', 'MAE: testing'])
        plt.xlabel('Iteration')
        plt.ylabel('Errors first deriv.')
        plt.show()

        plt.figure()
        plt.plot(np.diff(self.rmse_arr, n=2))
        plt.plot(np.diff(self.rmse_t_arr, n=2))
        plt.plot(np.diff(self.mae_t_arr, n=2))
        plt.grid()
        plt.legend(['Training', 'Testing', 'MAE: testing'])
        plt.xlabel('Iteration')
        plt.ylabel('Errors second deriv.')
        plt.show()

        plt.figure()
        plt.plot(self.pu_dev)
        plt.plot(self.qi_dev)
        plt.grid()
        plt.legend(['Pu', 'Qi'])
        plt.xlabel('Iteration')
        plt.ylabel('Dev from ortho')
        plt.show()


    def save_svd_params(self, svd_path):
        np.save('{}/bu.npy'.format(svd_path), self.bu)
        np.save('{}/bi.npy'.format(svd_path), self.bi)
        np.save('{}/pu.npy'.format(svd_path), self.pu)
        np.save('{}/qi.npy'.format(svd_path), self.qi)

        np.save('{}/pu_dev.npy'.format(svd_path), self.pu_dev)
        np.save('{}/qi_dev.npy'.format(svd_path), self.qi_dev)
        np.save('{}/rmse_arr.npy'.format(svd_path), self.rmse_arr)
        np.save('{}/rmse_t_arr.npy'.format(svd_path), self.rmse_t_arr)
        np.save('{}/mae_t_arr.npy'.format(svd_path), self.mae_t_arr)

    def load_svd_params(self, svd_path, mu):
        self.mu = mu
        self.bu = np.load('{}/bu.npy'.format(svd_path))
        self.bi = np.load('{}/bi.npy'.format(svd_path))
        self.pu = np.load('{}/pu.npy'.format(svd_path))
        self.qi = np.load('{}/qi.npy'.format(svd_path))

        self.pu_dev = np.load('{}/pu_dev.npy'.format(svd_path))
        self.qi_dev = np.load('{}/qi_dev.npy'.format(svd_path))
        self.rmse_arr = np.load('{}/rmse_arr.npy'.format(svd_path))
        self.rmse_t_arr = np.load('{}/rmse_t_arr.npy'.format(svd_path))
        self.mae_t_arr = np.load('{}/mae_t_arr.npy'.format(svd_path))

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def deviation_from_ortho(np.ndarray[np.double_t, ndim=2] M, size=1000):
    cdef np.ndarray[np.double_t, ndim=2] mask
    cdef np.ndarray[np.double_t, ndim=2] extract
    mask = np.ones((size, size)) - np.eye(size)
    extract = M[0:size, :]
    return np.abs(mask*(np.dot(extract, extract.transpose()))).sum()/size/size


print('Hello')
if __name__ == '__main__':
    print('Hello')
