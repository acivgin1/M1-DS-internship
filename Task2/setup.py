from setuptools import setup

requirements = [
    'numpy',
    'pandas',
    'scikit-learn',
    'matplotlib',
    'scipy'
]

setup(
    name='ds_internship_task2',
    version='0.1',
    url='https://github.com/acivgin1/M1-DS-internship',
    description='Testing standard classifiers on titanic dataset',
    entry_points={'console_scripts': ['run-all=ds_internship_task2.command_line:main']},
    author='Amar Civgin',
    author_email='amar.civgin@gmail.com',
    packages=['ds_internship_task2'],
    install_requires=requirements
)