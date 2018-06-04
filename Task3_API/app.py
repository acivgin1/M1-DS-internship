import pandas as pd
from flask import Flask, jsonify, request
from ds_movie_recommender.movie_recommender import SmartQi

DATA_PATH = '/home/acivgin/PycharmProjects/M1-DS-internship/Task3/Data'
SMARTQI = SmartQi(DATA_PATH)

app = Flask(__name__)


@app.route('/api/recommendations', methods=['GET', 'POST'])
def get_tasks():
    content = request.get_json(force=True)
    number_of_keys = len(content.keys())
    if not 'movie_id_list' in content.keys():
        return 'movie_id_list attribute is missing.'
    if number_of_keys == 2:
        if not 'movie_rating_list' in content.keys() and not 'number_of_movies' in content.keys():
            return 'movie_ratings or number_of_movies attributes is missing.'
    if number_of_keys == 3:
        if not 'movie_rating_list' in content.keys():
            return 'movie_ratings attributes is missing.'
        if not 'number_of_movies' in content.keys():
            return 'number_of_movies attributes is missing.'

    movie_id_list = content['movie_id_list']
    movie_rating_list = None
    number_of_movies = 10
    if 'movie_rating_list' in content.keys():
        movie_rating_list = content['movie_rating_list']
    if 'number_of_movies' in content.keys():
        number_of_movies = content['number_of_movies']

    recommendation = SMARTQI.give_n_recommendations(movie_id_list, movie_rating_list, number_of_movies, verbose=False)

    return recommendation.to_json()


@app.route('/api/movies', methods=['GET'])
def get_movie_list():
    return SMARTQI.movie_list.to_json(orient='values')


if __name__ == '__main__':
    smartqi = SmartQi(DATA_PATH)

    app.run()
