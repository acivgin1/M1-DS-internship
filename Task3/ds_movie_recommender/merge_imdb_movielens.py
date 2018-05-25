import os
import re

import pandas as pd
import numpy as np

cur_path = os.path.dirname(__file__)
data_path = os.path.relpath('../Data', cur_path)

imdb_path = '{}/imdb_movie_information'.format(data_path)
movielens_path = '{}/ratings_information'.format(data_path)

# imdb_df = pd.read_csv('{}/movie_metadata.csv'.format(imdb_path), usecols=[11, 17])
imdb_df = pd.read_csv('{}/title.basics.tsv'.format(imdb_path), usecols=[0, 3], sep='\t')
imdb_df['originalTitle'] = imdb_df['originalTitle'].str.strip()
imdb_df['imdbId'] = imdb_df['tconst'].apply(lambda x: int(re.findall('\\d{7}', x)[0]))

del imdb_df['tconst']
#
# imdb_df['movie_title'] = imdb_df['movie_title'].str.strip()
# imdb_df['imdbId'] = imdb_df['movie_imdb_link'].apply(lambda x: int(re.findall('\\d{7}', x)[0]))
#
# del imdb_df['movie_imdb_link']

# imdb_df.to_csv('{}/name_link.csv'.format(imdb_path), index=False)
imdb_df.set_index('imdbId', inplace=True)

movielens_df = pd.read_csv('{}/links.csv'.format(movielens_path), usecols=[0, 1], index_col=1)


joined_df = imdb_df.join(movielens_df, how='inner')
joined_df.set_index('movieId', inplace=True)
joined_df.sort_index(inplace=True)

joined_df.to_csv('{}/imdb_movielens.csv'.format(data_path))