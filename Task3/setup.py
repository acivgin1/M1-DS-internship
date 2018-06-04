from setuptools import setup
from Cython.Build import cythonize

requirements = [
    'numpy',
    'pandas',
    'matplotlib',
    'scipy',
    'tqdm',
    'cython'
]


setup(
    name='ds_movie_recommender',
    version='1.0',
    description='A movie recommender algorithm based around sparse SVD',
    author='Amar Civgin',
    author_email='amar.civgin@gmail.com',
    packages=['ds_movie_recommender'],
    install_requires=requirements,
    ext_modules=cythonize("ds_movie_recommender/svd_clustering.pyx")
)
