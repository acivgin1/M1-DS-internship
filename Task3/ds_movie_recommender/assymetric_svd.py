import numpy as np
from tqdm import tqdm


def svd_train(R, V, k_order=5, gamma=0.02, beta=0.05, num_of_iters=50, print_state=True):
    '''
    performs stochastic gradient descent on SVD algorithm

    :param R: a sparse COOrdinate matrix, list of (i, j, value) of Rating data
    :param V: a sparse COOrdinate matrix, list of (i, j, value) for Validation
    :param k_order: k-th order approximation
    :param gamma: learning rate
    :param beta: regularization parameter
    :param num_of_iters: number of iterations
    :param print_state: print_state errors
    :return: mu, bu, bi, pu, qi
    '''

    mean = 0
    std_dev = 0.1
    one_minus_gb = 1 - gamma * beta

    n_users = max(R.row.max(), V.row.max())+1
    n_items = max(R.col.max(), V.row.max())+1

    mu = (R.data.sum()/R.data.shape)[0]

    bu = np.zeros(n_users, dtype=np.double)
    bi = np.zeros(n_items, dtype=np.double)

    pu = np.random.normal(mean, std_dev, (n_users, k_order))
    qi = np.random.normal(mean, std_dev, (n_items, k_order))

    rmse_arr = []
    rmse_t_arr = []

    for iteration in tqdm(range(num_of_iters)):
        cum_error = 0
        for u, i, r in zip(R.row, R.col, R.data):
            error = r - (mu + bu[u] + bi[i] + np.dot(pu[u], qi[i]))
            cum_error += error * error

            error *= gamma

            bu[u] += gamma*(error - beta * bu[u])
            bi[i] += gamma*(error - beta * bi[i])

            puu = pu[u]
            pu[u] *= one_minus_gb
            pu[u] += error*qi[i]

            qi[i] *= one_minus_gb
            qi[i] += error*puu

            # for k in range(k_order):
            #     puk = pu[u, k]
            #
            #     pu[u, k] *= one_minus_gb
            #     pu[u, k] += error*qi[i, k]
            #
            #     qi[i, k] *= one_minus_gb
            #     qi[i, k] += error*puk

        cum_t_error = 0
        for u, i, r in zip(V.row, V.col, V.data):
            r_hat = svd_predict(u, i, pu, qi, mu, bu, bi)
            error = (r - r_hat)**2

            cum_t_error += error

        rmse = np.sqrt(cum_error/R.data.shape[0])
        rmse_t = np.sqrt(cum_t_error/V.data.shape[0])

        rmse_arr.append(rmse)
        rmse_t_arr.append(rmse_t)
        if print_state:
            print('epoch: {} error: {} t_error: {}'.format(iteration, rmse, rmse_t))
    return mu, bu, bi, pu, qi, rmse_arr, rmse_t_arr


def svd_predict(user_id, movie_id, pu, qi, mu=0, bu=0, bi=0):
    '''
    Returns predicted rating of user with user_id for a movie with movie_id

    :param user_id: integer
    :param movie_id: integer
    :param pu: estimated pu matrix
    :param qi: estimated qi matrix
    :param mu: mean of value matrix
    :param bu: user bias vector
    :param bi: item bias vector
    :return: r_ui predicted rating of user for item i
    '''
    if isinstance(bu, np.ndarray) and isinstance(bi, np.ndarray):
        return mu + bu[user_id] + bi[movie_id] + np.dot(pu[user_id], qi[movie_id])
    else:
        return mu + np.dot(pu[user_id], qi[movie_id])


def main():
    pass


if __name__ == '__main__':
    main()
