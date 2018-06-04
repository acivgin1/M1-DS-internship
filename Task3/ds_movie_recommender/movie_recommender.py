import numpy as np
import pandas as pd


class SmartQi:
    def __init__(self, data_path, read_from_svd_cluster=False):
        self.svd_path = '{}/SVD_path'.format(data_path)
        self.mu, self.bi, self.qi, self.qi_norm = None, None, None, None

        if read_from_svd_cluster:
            self.load_from_svd_cluster()
            self.save_smart_qi(qi_norm_to_bool=True)
        else:
            self.load_smart_qi()

        self.movie_list = pd.read_csv('{}/ratings_information/movies.csv'.format(data_path), index_col=0, usecols=[0, 1])

    def remove_zero_rating_movies(self, movie_nonzero):
        all_movie_ids = np.arange(0, self.qi.shape[0])
        zero_movie_ids = np.setxor1d(all_movie_ids, movie_nonzero, assume_unique=True)
        self.qi[zero_movie_ids] = 0
        return zero_movie_ids

    def load_smart_qi(self):
        loadz = np.load('{}/qi_params.npz'.format(self.svd_path))
        self.mu = loadz['mu']
        self.bi = loadz['bi']
        self.qi = loadz['qi']
        self.qi_norm = loadz['qi_norm']

    def load_from_svd_cluster(self, movie_nonzero):
        loadz = np.load('{}/svd_params.npz'.format(self.svd_path))
        self.mu = loadz['mu']
        self.bi = loadz['bi']
        self.qi = loadz['qi']

        self.remove_zero_rating_movies(movie_nonzero)

        qi_norm = np.linalg.norm(self.qi, axis=1)   # we need to norm the matrix qi, to save time
        self.qi_norm = qi_norm.copy()

        qi_norm[np.argwhere(qi_norm == 0)] = 1      # we don't want to divide with zero, but we need to know the zeros
        self.qi = self.qi / qi_norm.reshape((-1, 1))
        return

    def save_smart_qi(self, qi_norm_to_bool):
        if qi_norm_to_bool:
            self.qi_norm = self.qi_norm != 0
        np.savez('{}/qi_params.npz'.format(self.svd_path),
                 mu=self.mu,
                 bi=self.bi,
                 qi=self.qi,
                 qi_norm=self.qi_norm)

    def give_n_recommendations(self, movie_id_list, movie_rating_list=None, n=10, verbose=False):
        if not np.isin(movie_id_list, self.movie_list.index).all():
            missing_ids = np.setdiff1d(movie_id_list, self.movie_list.index)
            print('Missing ids are:\n{}'.format(missing_ids))
            movie_id_list = np.intersect1d(movie_id_list, self.movie_list.index)
            if movie_id_list.size == 0:
                print('There are no movies to search for.')
                return np.array([])

        if verbose:
            print(self.movie_list.loc[movie_id_list])

        movie_id_list = np.array(movie_id_list)
        if movie_rating_list is not None:
            movie_rating_list = np.array(movie_rating_list)

        recommended_movie_id_list = self.top_n_recommendations(movie_id_list, movie_rating_list, n + movie_id_list.size)
        recommended_movie_id_list = np.setdiff1d(recommended_movie_id_list, movie_id_list)

        if verbose:
            recommended_movies = self.movie_list.reindex(recommended_movie_id_list[:n]).dropna()
            print(recommended_movies.iloc[:n])
        return recommended_movie_id_list[:n]

    def top_n_recommendations(self, movie_id_list, movie_rating_list, n):
        pu = self.reduce_movie_vector(movie_id_list - 1, movie_rating_list)
        cosine = np.dot(pu, self.qi.transpose())
        cosine[np.argwhere(self.qi_norm == 0)] = -1     # since a score of -1 is practically impossible to reach
        return cosine.argsort()[-n:][::-1] + 1

    def reduce_movie_vector(self, movie_id_list, movie_rating_list):
        ru = np.zeros(self.qi.shape[0], dtype=np.double)
        if movie_rating_list is None:
            ru[movie_id_list] = self.mu + self.bi[movie_id_list]
        else:
            ru[movie_id_list] = movie_rating_list
        pu = np.dot(ru, self.qi)
        return pu / np.linalg.norm(pu)
