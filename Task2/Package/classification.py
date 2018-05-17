import warnings
import collections
from math import log10

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model, svm, naive_bayes
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

TITLES_FINAL = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']
AGE_INTERVALS = [6, 18, 32, 52, 72, 100]
DATASET_IDS = [17, 14, 8, 13]

# Classifier = collections.namedtuple('Classifier', 'model name params data_format_params dataset_params')
Classifier_result = collections.namedtuple('Classifier_test_result', 'classifier accuracy prediction')
Classifier_cv_result = collections.namedtuple('Classifier_cv_result', 'classifier f1_score')


class Classifier:
    def __init__(self, model, name, params, data_format_params, dataset_params):
        self.model = model
        self.model_cv = None
        self.name = name
        self.params = params
        self.data_format_params = data_format_params
        self.dataset_params = dataset_params


def age_to_group(age):
    return float(next(x[0] + 1 for x in enumerate(AGE_INTERVALS) if x[1] >= age))


def get_training_data(filename, scale_data, delete_fam_size, group_by_age):
    df = pd.read_csv('{}_training.csv'.format(filename), na_values='')

    del df['surname'], df['name'], df['sex']

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle data by sampling 1/1 fraction of dataframe

    labels = df['survived'].as_matrix()
    del df['survived']
    df['title'] = df['title'].apply(lambda x: TITLES_FINAL.index(x) + 1)

    if group_by_age:
        df['age'] = df['age'].apply(age_to_group)

    if scale_data:
        if not group_by_age:
            df['age'] = preprocessing.scale(df['age'])
        if not delete_fam_size:
            df['fam_size'] = preprocessing.scale(df['fam_size'])

    data = df.as_matrix()
    return data, labels


def get_all_data(filename, scale_data, delete_fam_size, group_by_age):
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


# gets classifier tuple, finds and sets params (model parameters) and returns classifier_cv_result
def tune_classifier_params(classifier, data, labels):
    if classifier.model_cv:
        classifier.model_cv.fit(data, labels)
        classifier.params = classifier.model_cv.best_params_
        classifier_cv_result = Classifier_cv_result(classifier=classifier,
                                                    f1_score=classifier.model_cv.best_score_)
    else:
        classifier.model.fit(data, labels)
        model_prediction = classifier.model.predict(data)
        f1_score = precision_recall_fscore_support(labels, model_prediction, average='binary')

        classifier.params = classifier.model.get_params()
        classifier_cv_result = Classifier_cv_result(classifier=classifier,
                                                    f1_score=f1_score[2])
    return classifier_cv_result


# gets list of classifiers and param_grids,
# then finds the best data_format_params for classifiers and returns classifier_cv_result
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


# filename = 'Data/Datasets/Titanic_{}'.format(i)
# gets list of classifiers and param_grids and finds best dataset_params
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


def dataset_params_to_dataset_id(dataset_params):
    mean, std_dev_scale, better_title = dataset_params
    dataset_id = 10*mean + 10*std_dev_scale + better_title + 1
    return int(dataset_id)


# gets classifier tuple, tests model and return classifier_result
def test_model(classifier, data, labels, data_t, labels_t):
    classifier.model.fit(data, labels)

    model_prediction = classifier.model.predict(data)
    model_prediction_t = classifier.model.predict(data_t)
    model_prediction_prob = classifier.model.predict_proba(data_t)

    model_acc = accuracy_score(labels, model_prediction)
    model_acc_t = accuracy_score(labels_t, model_prediction_t)

    classifier_result = Classifier_result(classifier=classifier,
                                          accuracy=(model_acc, model_acc_t),
                                          prediction=(model_prediction, model_prediction_t, model_prediction_prob))
    return classifier_result


def test_models(classifiers, data, labels, data_t, labels_t):
    classifier_results = []
    for classifier in classifiers:
        classifier_result = test_model(classifier, data, labels, data_t, labels_t)
        classifier_results.append(classifier_result)
    return classifier_results


def get_classifiers_and_param_grids():
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
    names = ['Decision tree', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Logistic Regression']

    sparse_param_grids = [dec_tree_params, gnb_params, svm_params, logreg_params]

    classifiers = [Classifier(model, name, None, None, None) for model, name in zip(models, names)]

    return classifiers, sparse_param_grids


def generate_param_grid(model_params, num, use_logspace):
    param_grid = {}
    for key, value in model_params.items():
        if use_logspace:
            start = log10(value) - 0.5
            stop = start + 1
            param_grid[key] = np.logspace(start, stop, num)
        else:
            start = value - int(num/2)
            start = start if start > 0 else 1
            stop = value + int(num/2)
            param_grid[key] = np.linspace(start, stop, num)
    return param_grid


def fine_tune_classifier_params(classifier_cv_result, filename):
    data_id = dataset_params_to_dataset_id(classifier_cv_result.classifier.dataset_params)
    filename = '{}_{}'.format(filename, data_id)

    scale_data, delete_fam_size, group_by_age = classifier_cv_result.classifier.data_format_params
    data, labels = get_training_data(filename, scale_data, delete_fam_size, group_by_age)

    use_logspace = classifier_cv_result.classifier.name != 'Decision tree'
    param_grid = generate_param_grid(classifier_cv_result.classifier.params, 5, use_logspace)

    classifier = classifier_cv_result.classifier
    classifier.cv_model = GridSearchCV(classifier.model,
                                       param_grid=param_grid,
                                       cv=4,
                                       scoring='f1',
                                       n_jobs=3)

    classifier_cv_result = tune_classifier_params(classifier, data, labels)
    return classifier_cv_result


def main(fine_tune_iterations=1):
    classifiers, param_grids = get_classifiers_and_param_grids()
    for classifier, param_grid in zip(classifiers, param_grids):
        if param_grid:
            classifier.model_cv = GridSearchCV(classifier.model,
                                               param_grid=param_grid,
                                               cv=4,
                                               scoring='f1',
                                               n_jobs=3)
        filename = 'Data/Datasets/Titanic'
        classifier_cv_result = tune_classifier_dataset_params(classifier, filename)

        if classifier.model_cv:
            for i in range(fine_tune_iterations):
                classifier_cv_result = fine_tune_classifier_params(classifier_cv_result, filename)

        fine_tuned_classifier = classifier_cv_result.classifier.model
        fine_tuned_classifier.set_params(**classifier_cv_result.classifier.params)

        test_filename = '{}_{}'.format(filename, dataset_params_to_dataset_id(classifier_cv_result.classifier.dataset_params))
        scale_data, delete_fam_size, group_by_age = classifier_cv_result.classifier.data_format_params
        data, labels, data_t, labels_t = get_all_data(test_filename, scale_data, delete_fam_size, group_by_age)

        classifier_result = test_model(classifier, data, labels, data_t, labels_t)
        print(classifier_result)


if __name__ == '__main__':
    main()
