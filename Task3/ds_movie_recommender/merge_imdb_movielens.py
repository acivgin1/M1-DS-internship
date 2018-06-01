import os
import re

import pandas as pd

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('../Data', cur_path)

imdb_path = '{}/imdb_movie_information'.format(data_path)
movielens_path = '{}/ratings_information'.format(data_path)

imdb_df = pd.read_csv('{}/title.basics.tsv'.format(imdb_path), usecols=[0, 2, 5], sep='\t', converters={'startYear': str})

imdb_df['primaryTitle'] = imdb_df['primaryTitle'].str.strip()
imdb_df['primaryTitle'] = imdb_df['primaryTitle'].str.cat(imdb_df['startYear'].tolist(), sep=' (')
imdb_df['primaryTitle'] = imdb_df['primaryTitle'] + ')'
del imdb_df['startYear']

imdb_df['imdbId'] = imdb_df['tconst'].apply(lambda x: int(re.findall('\\d{7}', x)[0]))
del imdb_df['tconst']

imdb_df.set_index('imdbId', inplace=True)

movielens_df = pd.read_csv('{}/links.csv'.format(movielens_path), usecols=[0, 1], index_col=1)


joined_df = imdb_df.join(movielens_df, how='inner')
joined_df.set_index('movieId', inplace=True)
joined_df.sort_index(inplace=True)

joined_df.to_csv('{}/imdb_movielens.csv'.format(data_path))