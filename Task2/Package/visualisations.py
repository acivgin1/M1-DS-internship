import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

TITLES_FINAL = ['Mr', 'Mrs', 'Ms', 'Master', 'Dr', 'Rev', 'Col', 'Sir', 'Lady']


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_age_histograms(pass_list, name):
    for i in range(1, 4):
        ax = plt.figure()
        legends = []
        for title in TITLES_FINAL[0:4]:
            age_list = np.array([x.age for x in pass_list if x.title == title and x.p_class == i and x.age > 0])
            legends.append('{} Class: {} Title: {}'.format(name, i, title))
            plt.hist(age_list, alpha=0.5, bins=15)
        plt.legend(legends)
        plt.title('{}. Class PDF'.format(i))
        plt.xlabel('Age [years]')
        plt.ylabel('Age frequency')
        ax.savefig('Data/Images/{}{}_pclass histogram.png'.format(i, name), bbox_inches='tight')


# plot age title
def plot_age_title(features, labels):
    fig = plt.figure()
    plt.xlabel('Age')
    plt.ylabel('Title')
    plt.yticks(range(0, len(TITLES_FINAL), 1), TITLES_FINAL)

    for elem, survived in zip(features, labels):
        if survived:
            plt.plot(elem[1], elem[0], 'ro')
        else:
            plt.plot(elem[1], elem[0], 'bv')

    fig.savefig('Data/Images/Age_Title.png', bbox_inches='tight')


def plot_age_title_class(features, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Passenger class')
    ax.set_xticks([3, 2, 1])

    ax.set_ylabel('Age')

    ax.set_zlabel('Title')
    ax.set_zticks(np.arange(0, len(TITLES_FINAL), 1))
    ax.set_zticklabels(TITLES_FINAL)

    for elem, survived in zip(features, labels):
        if survived:
            ax.scatter(elem[2], elem[1], elem[0], s=(1.2*elem[3]+2)**2, color='r', marker='o')
        else:
            ax.scatter(elem[2], elem[1], elem[0], s=(1.2*elem[3]+2)**2, color='b', marker='o')

    fig.savefig('Data/Images/Scatter.png', bbox_inches='tight')

    with open('Data/Pickles/myplot.pickle', 'wb') as fid:
        pickle.dump(ax, fid)

    with open('Data/Pickles/myplot.pickle', 'rb') as fid:
        ax = pickle.load(fid)
    plt.show()


def main():
    create_dir('Data/Images')
    create_dir('Data/Pickles')
    create_dir('Data/Dataframes')

    df = pd.read_csv('Data/Datasets/Titanic_1_training.csv', na_values='')
    df['died'] = np.ones((df.shape[0], 1), dtype=np.uint8) - df['survived'].as_matrix().reshape((-1, 1))
    df['ones'] = np.ones((df.shape[0], 1), dtype=np.uint8)

    age_intervals = [0, 1, 3, 6, 12, 18, 22, 27, 32, 37, 42, 47, 52, 57, 62, 72]

    df['age_interval'] = pd.cut(df.age, age_intervals)

    survive_title = df.groupby(['title']).agg({'survived': ['sum'], 'died': ['sum']})
    survive_sex = df.groupby(['sex']).agg({'survived': ['sum'], 'died': ['sum']})
    survive_class = df.groupby(['p_class']).agg({'survived': ['sum'], 'died': ['sum']})
    survive_age_sex = df.groupby(['age_interval', 'sex']).agg({'survived': ['sum'], 'died': ['sum']})
    survive_age_sex_class = df.groupby(['age_interval', 'sex', 'p_class']).agg({'survived': ['sum'], 'died': ['sum']})
    survive_age_title = df.groupby(['age_interval', 'title']).agg({'survived': ['sum'], 'died': ['sum']})
    survive_age_title_class = df.groupby(['age_interval', 'title', 'p_class']).agg({'survived': ['sum'],
                                                                                    'died': ['sum'],
                                                                                    'ones': ['sum']})

    survive_age_title_famsize = df.groupby(['age_interval', 'title', 'fam_size']).agg({'survived': ['sum'],
                                                                                       'died': ['sum'],
                                                                                       'ones': ['sum']})

    survive_age_title_class_famsize = df.groupby(['age_interval', 'title', 'p_class', 'fam_size']).agg({'survived': ['sum'],
                                                                                                        'died': ['sum'],
                                                                                                        'ones': ['sum']})

    survive_age_title_class_famsize.to_csv('Data/Dataframes/survive_age_title_class_famsize.csv')

    del df['surname'], df['name'], df['sex'], df['died'], df['ones'], df['age_interval']
    labels = df['survived'].as_matrix()
    del df['survived']

    df['title'] = df['title'].apply(lambda x: TITLES_FINAL.index(x))
    features = df.as_matrix()

    plot_age_title(features, labels)
    plot_age_title_class(features, labels)


if __name__ == '__main__':
    main()
