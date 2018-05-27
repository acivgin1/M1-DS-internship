import os
import pickle
import warnings
import itertools
from math import log10

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model, svm, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

warnings.filterwarnings('ignore')

TITLES_FINAL = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']
AGE_INTERVALS = [6, 18, 32, 52, 72, 100]
DATASET_IDS = [17, 14, 8, 13]
CLASSIFIER_NAMES = ['Decision tree', 'Gaussian Naive Bayes', 'Support Vector Machine', 'Logistic Regression']


class ClassifierResult:
    def __init__(self, classifier, accuracy, prediction):
        self.classifier = classifier
        self.accuracy = accuracy
        self.prediction = prediction

    def save(self, pickles_path, name):
        with open('{}/result_{}.pkl'.format(pickles_path, name), 'wb') as fod:
            pickle.dump(self, fod)

    @staticmethod
    def load(pickles_path, name):
        with open('{}/result_{}.pkl'.format(pickles_path, name), 'rb') as fid:
            obj = pickle.load(fid)
        return obj


class ClassifierCvResult:
    def __init__(self, classifier, f1_score):
        self.classifier = classifier
        self.f1_score = f1_score

    def save(self, pickles_path, name):
        with open('{}/cv_result_{}.pkl'.format(pickles_path, name), 'wb') as fod:
            pickle.dump(self, fod)

    @staticmethod
    def load(pickles_path, name):
        with open('{}/cv_result_{}.pkl'.format(pickles_path, name), 'rb') as fid:
            obj = pickle.load(fid)
        return obj


class Classifier:
    def __init__(self, model, name, model_params, data_format_params, dataset_params):
        self.model = model
        self.model_cv = None
        self.name = name
        self.model_params = model_params
        self.data_format_params = data_format_params
        self.dataset_params = dataset_params


# converts a continuous scalar to a categorical variable
def age_to_group(age):
    return float(next(x[0] + 1 for x in enumerate(AGE_INTERVALS) if x[1] >= age))


# takes filename and data_format_params and loads training data from disk
def get_training_data(filename, scale_data, delete_fam_size, group_by_age):
    df = pd.read_csv('{}_training.csv'.format(filename), na_values='')

    del df['surname'], df['name'], df['sex']

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data by sampling 1/1 fraction of dataframe

    labels = df['survived'].as_matrix()
    del df['survived']
    df['title'] = df['title'].apply(lambda x: TITLES_FINAL.index(x) + 1)

    if group_by_age:
        df['age'] = df['age'].apply(age_to_group).astype(float)

    if scale_data:
        if not group_by_age:
            df['age'] = preprocessing.scale(df['age'].astype(float))
        if not delete_fam_size:
            df['fam_size'] = preprocessing.scale(df['fam_size'].astype(float))

    data = df.as_matrix()
    return data, labels


# takes filename and data_format_params and loads training and test data from disk
def get_training_and_test_data(filename, scale_data, delete_fam_size, group_by_age):
    df1 = pd.read_csv('{}_training.csv'.format(filename), na_values='')
    df2 = pd.read_csv('{}_testing.csv'.format(filename), na_values='')

    df1_len = df1.shape[0]

    labels = df1['survived'].as_matrix()
    labels_t = df2['survived'].as_matrix()
    del df1['survived'], df2['survived']

    df = pd.concat([df1, df2], ignore_index=True)

    del df['surname'], df['name'], df['sex']
    if delete_fam_size:
        del df['fam_size']

    df['title'] = df['title'].apply(lambda x: TITLES_FINAL.index(x) + 1)

    if group_by_age:
        df['age'] = df['age'].apply(age_to_group)

    if scale_data:
        if not group_by_age:
            df['age'] = preprocessing.scale(df['age'])
        if not delete_fam_size:
            df['fam_size'] = preprocessing.scale(df['fam_size'])

    df = df.as_matrix()

    data = df[:df1_len]
    data_t = df[df1_len:]
    return data, labels, data_t, labels_t


# takes classifier tuple and data, cv_searches and sets model_params and returns classifier_cv_result
def tune_classifier_params(classifier, data, labels):
    if classifier.model_cv:
        classifier.model_cv.fit(data, labels)
        classifier.model_params = classifier.model_cv.best_params_
        classifier_cv_result = ClassifierCvResult(classifier=classifier,
                                                  f1_score=classifier.model_cv.best_score_)
    else:
        f1_score = cross_val_score(classifier.model, data, labels, scoring='f1', cv=4, n_jobs=3).mean()
        classifier.model_params = classifier.model.get_params()
        classifier_cv_result = ClassifierCvResult(classifier=classifier,
                                                  f1_score=f1_score)
    return classifier_cv_result


# takes list of classifiers and param_grids, cv_searches and sets data_format_params and returns classifier_cv_result
def tune_classifier_data_format_params(classifier, filename):
    best_f1_score = 0
    best_classifier_cv_result = None

    for scale_data in [False, True]:
        for delete_fam_size in [True, False]:
            for group_by_age in [False, True]:
                if scale_data and group_by_age and delete_fam_size:
                    continue

                data, labels = get_training_data(filename, scale_data, delete_fam_size, group_by_age)
                classifier_cv_result = tune_classifier_params(classifier, data, labels)

                if classifier_cv_result.f1_score > best_f1_score:
                    best_classifier_cv_result = classifier_cv_result
                    best_classifier_cv_result.classifier.data_format_params = (scale_data,
                                                                               delete_fam_size,
                                                                               group_by_age)
                    best_f1_score = classifier_cv_result.f1_score
    return best_classifier_cv_result


# takes list of classifiers and param_grids, cv searches and sets dataset_params and return classifier_cv_result
def tune_classifier_dataset_params(classifier, filename):
    best_f1_score = 0
    best_classifier_cv_result = None
    dataset_count = 1
    for mean in [False, True]:
        for std_dev_scale in [0, 0.2, 0.4, 0.6, 0.8]:
            for better_title in [False, True]:
                numbered_filename = '{}_{}'.format(filename, dataset_count)
                dataset_count += 1

                classifier_cv_result = tune_classifier_data_format_params(classifier, numbered_filename)

                if classifier_cv_result.f1_score > best_f1_score:
                    best_classifier_cv_result = classifier_cv_result
                    best_classifier_cv_result.classifier.dataset_params = (mean, std_dev_scale, better_title)

                    best_f1_score = classifier_cv_result.f1_score
    return best_classifier_cv_result


# converts dataset_params to integer representing that dataset id
def dataset_params_to_dataset_id(dataset_params):
    mean, std_dev_scale, better_title = dataset_params
    dataset_id = 10*mean + 10*std_dev_scale + better_title + 1
    return int(dataset_id)


# takes classifier tuple, tests model and returns classifier_result
def test_model(classifier, data, labels, data_t, labels_t):
    classifier.model.fit(data, labels)

    model_prediction = classifier.model.predict(data)
    model_prediction_t = classifier.model.predict(data_t)
    model_prediction_prob = classifier.model.predict_proba(data_t)

    model_acc = accuracy_score(labels, model_prediction)
    model_acc_t = accuracy_score(labels_t, model_prediction_t)

    classifier_result = ClassifierResult(classifier=classifier,
                                         accuracy=(model_acc, model_acc_t),
                                         prediction=(model_prediction, model_prediction_t, model_prediction_prob))
    return classifier_result


# creates list of classifier tuples and param_grids
def get_classifiers_and_param_grids():
    global CLASSIFIER_NAMES

    dec_tree_params = {'max_depth': np.arange(2, 10)}
    dec_tree_clas = tree.DecisionTreeClassifier()

    svm_params = {'C': [0.1, 0.3, 1, 3, 10, 100, 300],
                  'gamma': [0.003, 0.01, 0.03, 0.1]}
    svm_clas = svm.SVC(kernel='rbf', probability=True)

    gnb_params = {}
    gnb_clas = naive_bayes.GaussianNB()

    logreg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    logreg_clas = linear_model.LogisticRegression()

    models = [dec_tree_clas, gnb_clas, svm_clas, logreg_clas]

    sparse_param_grids = [dec_tree_params, gnb_params, svm_params, logreg_params]

    classifiers = [Classifier(model, name, None, None, None) for model, name in zip(models, CLASSIFIER_NAMES)]

    return classifiers, sparse_param_grids


# generates a new n-dim param_grid with n*num points around the last model_param n-dim point
def generate_param_grid(model_params, num, use_logspace, fraction):
    param_grid = {}
    for key, value in model_params.items():
        if use_logspace:
            start = log10(value) - 1/float(fraction)
            stop = start + 2/float(fraction)
            param_grid[key] = np.logspace(start, stop, num)
        else:
            start = value - int(num/fraction)
            start = start if start > 0 else 1
            stop = value + int(num/fraction)
            param_grid[key] = np.linspace(start, stop, num)
    return param_grid


def generate_weight_grid(weights, num_of_models):
    return [x for x in itertools.product(weights, repeat=num_of_models)]


# fine tunes a classifier using dataset_params and data_format_params from classifier_cv_result
# and returns classifier_result
def fine_tune_classifier_params(classifier_cv_result, fraction, filename):
    data_id = dataset_params_to_dataset_id(classifier_cv_result.classifier.dataset_params)
    filename = '{}_{}'.format(filename, data_id)

    scale_data, delete_fam_size, group_by_age = classifier_cv_result.classifier.data_format_params
    data, labels = get_training_data(filename, scale_data, delete_fam_size, group_by_age)

    # can't generate float param_grid when dealing with integer attributes
    use_logspace = classifier_cv_result.classifier.name != 'Decision tree'
    param_grid = generate_param_grid(classifier_cv_result.classifier.model_params, 5, use_logspace, fraction)

    classifier = classifier_cv_result.classifier
    classifier.cv_model = GridSearchCV(classifier.model,
                                       param_grid=param_grid,
                                       cv=4,
                                       scoring='f1',
                                       n_jobs=3)

    classifier_cv_result = tune_classifier_params(classifier, data, labels)
    return classifier_cv_result


def tune_classifiers(classifiers, param_grids, filename, pickles_path, fine_tune_iterations=1):
    fine_tuned_classifiers = []
    classifier_results = []
    classifier_cv_results = []

    for classifier, param_grid in zip(classifiers, param_grids):
        if param_grid:
            classifier.model_cv = GridSearchCV(classifier.model,
                                               param_grid=param_grid,
                                               cv=4,
                                               scoring='f1',
                                               n_jobs=3)

        classifier_cv_result = tune_classifier_dataset_params(classifier, filename)
        classifier_cv_result.save(pickles_path, classifier.name)

        if classifier.model_cv:
            for i in range(fine_tune_iterations):
                classifier_cv_result = fine_tune_classifier_params(classifier_cv_result, 2 ** (i + 1), filename)

        fine_tuned_classifier = classifier_cv_result.classifier.model
        fine_tuned_classifier.set_params(**classifier_cv_result.classifier.model_params)

        test_filename = '{}_{}'.format(filename,
                                       dataset_params_to_dataset_id(classifier_cv_result.classifier.dataset_params))
        scale_data, delete_fam_size, group_by_age = classifier_cv_result.classifier.data_format_params
        data, labels, data_t, labels_t = get_training_and_test_data(test_filename, scale_data, delete_fam_size,
                                                                    group_by_age)

        classifier_result = test_model(classifier, data, labels, data_t, labels_t)

        fine_tuned_classifiers.append(fine_tuned_classifier)
        classifier_cv_results.append(classifier_cv_result)
        classifier_results.append(classifier_result)

    return fine_tuned_classifiers, classifier_cv_results, classifier_results


def tune_voting_classifier(classifier_cv_results, filename, pickles_path):
    classifiers = []
    weights = []
    full_filename = '{}_{}'.format(filename, dataset_params_to_dataset_id((True, 0.2, False)))

    data, labels, data_t, labels_t = get_training_and_test_data(full_filename, True, False, True)

    for classifier_cv_result in classifier_cv_results:
        classifier_cv_result.classifier.data_format_params = (True, False, True)
        classifier_cv_result.classifier.dataset_params = (True, 0.2, False)

        if classifier_cv_result.classifier.name != 'Gaussian Naive Bayes':
            for i in range(2):
                classifier_cv_result = fine_tune_classifier_params(classifier_cv_result, 2 ** i, filename)

        classifiers.append(classifier_cv_result.classifier.model)
        classifier_result = test_model(classifier_cv_result.classifier, data, labels, data_t, labels_t)

        weights.append(classifier_result.accuracy[0])

    voting_model = VotingClassifier(estimators=[('dt', classifiers[0]),
                                                ('gnb', classifiers[1]),
                                                ('svm', classifiers[2]),
                                                ('lr', classifiers[3])
                                                ], voting='soft', weights=weights)
    voting_clas = Classifier(model=voting_model,
                             name='Voting Classifier',
                             model_params={'weights': voting_model.weights},
                             data_format_params=(True, False, True),
                             dataset_params=(True, 0.2, False))

    classifier_result = test_model(voting_clas, data, labels, data_t, labels_t)
    classifier_result.save(pickles_path, 'Voting Classifier')

    return classifier_result


def load_classifiers(filename, pickles_path):
    fine_tuned_classifiers = []
    classifier_cv_results = []
    classifier_results = []

    for name in CLASSIFIER_NAMES:
        classifier_cv_result = ClassifierCvResult.load(pickles_path, name)

        fine_tuned_classifier = classifier_cv_result.classifier.model
        fine_tuned_classifier.set_params(**classifier_cv_result.classifier.model_params)

        test_filename = '{}_{}'.format(filename,
                                       dataset_params_to_dataset_id(
                                           classifier_cv_result.classifier.dataset_params))
        scale_data, delete_fam_size, group_by_age = classifier_cv_result.classifier.data_format_params
        data, labels, data_t, labels_t = get_training_and_test_data(test_filename, scale_data, delete_fam_size,
                                                                    group_by_age)

        classifier_result = test_model(classifier_cv_result.classifier, data, labels, data_t, labels_t)

        fine_tuned_classifiers.append(fine_tuned_classifier)
        classifier_cv_results.append(classifier_cv_result)
        classifier_results.append(classifier_result)

    return fine_tuned_classifiers, classifier_cv_results, classifier_results


def print_classifier_cv_result(classifier_cv_result):
    print('{}: {} {} {}'.format(classifier_cv_result.classifier.name,
                                classifier_cv_result.classifier.model_params,
                                classifier_cv_result.classifier.data_format_params,
                                classifier_cv_result.classifier.dataset_params))


# this is a work in progress create_new_datasets method that does hyper model_params and model model_params
# tuning and testing of different models
def classifier_main(train_models=True, print_results=True, data_path=None):
    if data_path is None:
        cur_path = os.path.dirname(__file__)
        data_path = os.path.relpath('../Data', cur_path)

    filename = '{}/Datasets/Titanic'.format(data_path)
    pickles_path = '{}/Pickles'.format(data_path)

    classifiers, param_grids = get_classifiers_and_param_grids()
    if train_models:
        fine_tuned_classifiers, classifier_cv_results, classifier_results = tune_classifiers(classifiers=classifiers,
                                                                                             param_grids=param_grids,
                                                                                             filename=filename,
                                                                                             pickles_path=pickles_path)
        voting_classifier_result = tune_voting_classifier(classifier_cv_results, filename, pickles_path)
    else:
        fine_tuned_classifiers, classifier_cv_results, classifier_results = load_classifiers(filename, pickles_path)
        voting_classifier_result = ClassifierResult.load(pickles_path, 'Voting Classifier')

    if print_results:
        for res, res1 in zip(classifier_cv_results, classifier_results):
            print_classifier_cv_result(res)
            print(res1.accuracy)
            for elem in res1.prediction:
                print(elem)

        print(voting_classifier_result.classifier.name)
        print(voting_classifier_result.accuracy)
        for elem in voting_classifier_result.prediction:
            print(elem)


if __name__ == '__main__':
    classifier_main(False)
