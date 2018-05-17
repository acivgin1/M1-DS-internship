from setuptools import setup

requirements = [
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib'
]

setup(
    name='ds-internship-task2',
    version='0.1',
    url='https://github.com/acivgin1/M1-DS-internship',
    description='Testing standard classifiers on titanic dataset',
    author='Amar Civgin',
    author_email='amar.civgin@gmail.com',
    packages=['ds-internship-task2'],
    install_requires=requirements
)