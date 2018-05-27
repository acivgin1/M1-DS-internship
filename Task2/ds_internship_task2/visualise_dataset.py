import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

TITLES_FINAL = ['Mr', 'Mrs', 'Ms', 'Master', 'Dr', 'Rev', 'Col', 'Sir', 'Lady']

PARAMS_TO_GROUP_BY = ['age_interval', 'title', 'p_class', 'fam_size', 'sex']
PARAMS_LIST = [[0, 4], [0, 1], [0, 4, 2], [0, 1, 2], [0, 1, 2, 3]]

USED_DIRS_NAMES = ['Images', 'Pickles', 'Dataframes']


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_age_histograms(df, dataset_id, data_path):
    for p_class in range(1, 4):
        ax = plt.figure()
        legends = []
        grouped = df.groupby(['title', 'p_class'], sort=False)['age']

        for title in TITLES_FINAL[0:4]:
            age_list = grouped.get_group((title, p_class)).as_matrix()
            legends.append('Dataset_{} Class: {} Title: {}'.format(dataset_id, p_class, title))
            plt.hist(age_list, alpha=0.5, bins=15)
        plt.legend(legends)
        plt.title('{}. Class PDF'.format(p_class))
        plt.xlabel('Age [years]')
        plt.ylabel('Age frequency')
        ax.savefig('{}/Images/{}_pclass_histogram_{}.png'.format(data_path, p_class, dataset_id), bbox_inches='tight')


def plot_age_title(features, labels, data_path):
    fig = plt.figure()
    plt.xlabel('Age')
    plt.ylabel('Title')
    plt.yticks(range(0, len(TITLES_FINAL), 1), TITLES_FINAL)

    for elem, survived in zip(features, labels):
        if survived:
            plt.plot(elem[1], elem[0], 'ro')
        else:
            plt.plot(elem[1], elem[0], 'bv')

    fig.savefig('{}/Images/Age_Title.png'.format(data_path), bbox_inches='tight')


def plot_age_title_class(features, labels, data_path):
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

    fig.savefig('{}/Images/Scatter.png'.format(data_path), bbox_inches='tight')

    with open('{}/Pickles/age_title_class.pickle'.format(data_path), 'wb') as fid:
        pickle.dump(ax, fid)


def load_and_plot_pickle(filename):
    with open(filename, 'rb') as fid:
        pickle.load(fid)
        plt.show()


def plot_features_and_labels(df, data_path, dataset_id):
    del df['surname'], df['name'], df['sex'], df['died'], df['age_interval']
    labels = df['survived'].as_matrix()
    del df['survived']

    df['title'] = df['title'].apply(lambda x: TITLES_FINAL.index(x))
    features = df.as_matrix()

    plot_age_title(features, labels, data_path)

    plot_age_title_class(features, labels, data_path)


def group_survivors_by_params(df, params_list, data_path):
    grouped_df = df.groupby(params_list).agg({'survived': ['sum'], 'died': ['sum']})

    if data_path:
        filename = '{}/survive'.format(data_path)
        for param in params_list[:-1]:
            filename = '{}_{}'.format(filename, param)
        filename = '{}_{}.csv'.format(filename, params_list[-1])
        grouped_df.to_csv(filename)


def create_survivor_groups(df, data_path):
    age_intervals = [0, 1, 3, 6, 12, 18, 22, 27, 32, 37, 42, 47, 52, 57, 62, 72]

    df['age_interval'] = pd.cut(df.age, age_intervals)
    for params in PARAMS_TO_GROUP_BY:
        group_survivors_by_params(df=df,
                                  params_list=[params],
                                  data_path='{}/{}'.format(data_path, USED_DIRS_NAMES[-1]))

    for params_list_ids in PARAMS_LIST:
        params_list = [PARAMS_TO_GROUP_BY[param_id] for param_id in params_list_ids]
        group_survivors_by_params(df=df,
                                  params_list=params_list,
                                  data_path='{}/{}'.format(data_path, USED_DIRS_NAMES[-1]))


def create_visualisations(dataset_id=14, data_path=None, show_last_plot=False):
    if data_path is None:
        cur_path = os.path.dirname(__file__)
        data_path = os.path.relpath('../Data', cur_path)

    for used_dirs_name in USED_DIRS_NAMES:
        create_dir('{}/{}'.format(data_path, used_dirs_name))

    df = pd.read_csv('{}/Datasets/Titanic_{}_training.csv'.format(data_path, dataset_id), na_values='')
    df['died'] = np.ones((df.shape[0], 1), dtype=np.uint8) - df['survived'].as_matrix().reshape((-1, 1))

    plot_age_histograms(df, dataset_id=dataset_id, data_path=data_path)
    create_survivor_groups(df, data_path)
    plot_features_and_labels(df, data_path, dataset_id)
    if show_last_plot:
        plt.show()


if __name__ == '__main__':
    create_visualisations()
