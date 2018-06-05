import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from ds_movie_recommender.movie_recommender import SmartQi

DATA_PATH = '/home/acivgin/PycharmProjects/M1-DS-internship/Task3/Data'
SMARTQI = SmartQi(DATA_PATH)
LINKS = pd.read_csv('{}/ratings_information/links.csv'.format(DATA_PATH))

app = Flask(__name__)


@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    global LINKS, SMARTQI

    movie_id_list = request.args.get('movie_id_list').split(',')
    movie_id_list = list(map(int, movie_id_list))

    movie_rating_list = request.args.get('movie_rating_list')
    if movie_rating_list is not None:
        movie_rating_list = movie_rating_list.split(',')
        movie_rating_list = list(map(int, movie_rating_list))
    print(movie_rating_list)

    recommendation = SMARTQI.give_n_recommendations(movie_id_list, movie_rating_list, verbose=True)
    recommendation = SMARTQI.movie_list.reindex(recommendation).dropna()
    return pd.Series(recommendation['title']).to_json(orient='values')


@app.route('/api/recommendations', methods=['GET', 'POST'])
def get_recommendations_json():
    global LINKS, SMARTQI
    content = request.get_json(force=True)
    number_of_keys = len(content.keys())

    if not 'movie_id_list' in content.keys():
        return 'movie_id_list attribute is missing.'
    if not 'moviedb' in content.keys():
        return 'moviedb attribute is missing.'

    if number_of_keys == 3:
        if not 'movie_rating_list' in content.keys() and not 'number_of_movies' in content.keys():
            return 'movie_ratings or number_of_movies attributes is missing.'
    if number_of_keys == 4:
        if not 'movie_rating_list' in content.keys():
            return 'movie_ratings attributes is missing.'
        if not 'number_of_movies' in content.keys():
            return 'number_of_movies attributes is missing.'

    movie_id_list = content['movie_id_list']

    if content['moviedb'] == 'tmdb':
        LINKS.set_index('tmdbId', inplace=True, drop=False)
        movie_id_list = LINKS.loc[movie_id_list]['movieId'].tolist()

    if content['moviedb'] == 'imdb':
        LINKS.set_index('imdbId', inplace=True, drop=False)
        movie_id_list = LINKS.loc[movie_id_list]['movieId'].tolist()

    movie_rating_list = None
    number_of_movies = 10
    if 'movie_rating_list' in content.keys():
        movie_rating_list = content['movie_rating_list']
    if 'number_of_movies' in content.keys():
        number_of_movies = content['number_of_movies']

    recommendation = SMARTQI.give_n_recommendations(movie_id_list, movie_rating_list, number_of_movies, verbose=True)
    print(recommendation)

    LINKS.set_index('movieId', inplace=True, drop=False)
    if content['moviedb'] == 'tmdb':
        recommendation = LINKS.loc[recommendation]['tmdbId'].astype(np.uint16).tolist()
    if content['moviedb'] == 'imdb':
        recommendation = LINKS.loc[recommendation]['imdbId'].astype(np.uint16).tolist()

    return pd.Series(recommendation).to_json(orient='values')


@app.route('/api/movies', methods=['GET'])
def get_movie_list():
    return SMARTQI.movie_list.to_json()


@app.route('/')
def help():
    help_string = 'You seem to be lost. If you want to get the movie list, go to /api/movies.' \
                  '\nIf you want to get movie recommedations go to /api/recommendations and give me a json file' \
                  'with attributes movie_id_list, moviedb, and optionally movie_rating_list, number_of_movies.'
    return(help_string)


if __name__ == '__main__':
    app.run()
