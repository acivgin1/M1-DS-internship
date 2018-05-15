import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model, svm, naive_bayes
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']
age_intervals = [6, 18, 32, 52, 72, 100]


def age_to_group(age):
    return float(next(x[0]+1 for x in enumerate(age_intervals) if x[1] >= age))


def cross_validation_search(model, model_name, data, labels):
    model.fit(data, labels)
    if hasattr(model, 'best_params_'):
        print("Best parameters set found on development set:")
        print(model.best_params_)
        print('Best score is: {}'.format(model.best_score_))
    else:
        model_pred = model.predict(data)
        print('Best score is: {}'.format(accuracy_score(labels, model_pred)))


def testing_models(model, model_name, data, labels, t_data, t_labels, final=False):
    model.fit(data, labels)

    model_pred = model.predict(data)
    model_pred_t = model.predict(t_data)
    model_pred_t_prob = model.predict_proba(t_data)

    model_acc = np.equal(model_pred, labels).sum() / labels.shape[0]
    model_acc_t = np.equal(model_pred_t, t_labels).sum() / t_labels.shape[0]

    model_precision = average_precision_score(t_labels, model_pred_t)
    f1_score = precision_recall_fscore_support(t_labels, model_pred_t, average='micro')

    print('{}: acc: {}, t_acc: {}, prec: {}, f1: {}'.format(model_name, model_acc, model_acc_t, model_precision, f1_score))
    print('Labels: {}, prediction: {}, prediction probability: {}'.format(t_labels, model_pred_t, model_pred_t_prob))


def get_data(filename, preprocess, delete_fam_size, group_by_age, training):
    df1 = pd.read_csv('{}_training.csv'.format(filename), na_values='')
    del df1['surname'], df1['name'], df1['sex']
    if delete_fam_size:
        del df1['fam_size']

    t_data = None
    t_labels = None
    df1_len = -1

    if training:
        df = df1.sample(frac=1).reset_index(drop=True)
        labels = df['survived'].as_matrix()
        del df['survived']
    else:
        df2 = pd.read_csv('{}_testing.csv'.format(filename), na_values='')

        del df2['surname'], df2['name'], df2['sex']
        if delete_fam_size:
            del df2['fam_size']

        df1_len = df1.shape[0]

        labels = df1['survived'].as_matrix()
        t_labels = df2['survived'].as_matrix()
        del df1['survived'], df2['survived']

        df = pd.concat([df1, df2], ignore_index=True)

    df['title'] = df['title'].apply(lambda x: titles_final.index(x)+1)

    if group_by_age:
        df['age'] = df['age'].apply(age_to_group)

    if preprocess:
        if not group_by_age:
            df['age'] = preprocessing.scale(df['age'])
        if not delete_fam_size:
            df['fam_size'] = preprocessing.scale(df['fam_size'])

    df = df.as_matrix()

    if training:
        data = df
    else:
        data = df[:df1_len]
        t_data = df[df1_len:]
    return data, labels, t_data, t_labels


def test_hypothesis(filename, use_cross_validation, training):
    dec_parameters = {'max_depth': np.arange(3, 10)}
    dectree = tree.DecisionTreeClassifier()

    svm_parameters = {'C': [0.1, 0.3, 1, 3, 10, 100, 300],
                      'gamma': [0.003, 0.01, 0.03, 0.1]}
    svmclas = svm.SVC(kernel='rbf')

    gnbclas = naive_bayes.GaussianNB()

    logreg_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    logreg = linear_model.LogisticRegression(C=1e5)

    models = [dectree, svmclas, gnbclas, logreg]
    model_names = ['Decision tree', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Logistic Regression']
    parameters = [dec_parameters, svm_parameters, {}, logreg_parameters]

    if use_cross_validation:
        for delete_fam_size in [True, False]:
            for group_by_age in [False, True]:
                for preprocess in [False, True]:
                    if preprocess and group_by_age and delete_fam_size:
                        continue
                    print('')
                    print('scale[{}]'.format('on' if preprocess else 'off'), end=', ')
                    print('fam_size[{}]'.format('on' if not delete_fam_size else 'off'), end=', ')
                    print('group_age[{}]'.format('on' if group_by_age else 'off'), end='.\n')

                    for model, model_name, model_parameters in zip(models, model_names, parameters):
                        if use_cross_validation and not model_name == model_names[2]:
                            model = GridSearchCV(model,
                                                 param_grid=model_parameters,
                                                 cv=4,
                                                 scoring='f1',
                                                 n_jobs=8)
                        print('\n{}'.format(model_name))
                        data, labels, _, _ = get_data(filename,
                                                      preprocess,
                                                      delete_fam_size,
                                                      group_by_age,
                                                      training)
                        cross_validation_search(model, model_name, data, labels)


def main(log_to_file, sparse_cv_search, final_run=False):
    if log_to_file:
        orig_stdout = sys.stdout
        f = open('final_results.txt', 'w')
        sys.stdout = f

    if sparse_cv_search:
        i = 1
        for mean in [True, False]:
            for std_dev_scale in [0, 0.2, 0.4, 0.6, 0.8]:
                for better_title in [False, True]:
                    filename = 'Data/Datasets/Titanic_{}'.format(i)
                    print('\n############################################')
                    print('mean[{}], std_dev_scale[{}], better_title[{}]\n'.format(mean, std_dev_scale, better_title))
                    test_hypothesis(use_cross_validation=True, filename=filename, training=True)
                    i += 1
    else:
        dataset_ids = [17, 14, 8, 13]
        dectree = tree.DecisionTreeClassifier()         # depth: 9
        svmclas = svm.SVC(kernel='rbf')                 # C: 3, gamma: 0.01
        gnbclas = naive_bayes.GaussianNB()              # no params
        logreg = linear_model.LogisticRegression()      # C: 10

        models = [dectree, svmclas, gnbclas, logreg]
        model_names = ['Decision tree', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Logistic Regression']

        parameters = [9,
                      {'C': np.arange(3.5, 4, step=0.01), 'gamma': np.arange(0.04, 0.05, 0.001)},
                      None,
                      {'C': np.arange(2.1, 2.3, 0.001)}]
        hyper_parameters = [[True, False, True],
                            [False, False, True],
                            [False, False, False],
                            [False, False, False]]

        for model, name, params, hyper_params, data_id in zip(models, model_names, parameters, hyper_parameters, dataset_ids):
            preprocess, delete_fam_size, group_by_age = hyper_params[0], hyper_params[1], hyper_params[2]
            filename = 'Data/Datasets/Titanic_{}'.format(data_id)

            data, labels, t_data, t_labels = get_data(filename,
                                                      preprocess,
                                                      delete_fam_size,
                                                      group_by_age,
                                                      sparse_cv_search)
            if not final_run:
                if type(params) is dict:
                    model = GridSearchCV(model, param_grid=params, cv=4, scoring='f1', n_jobs=4)
                cross_validation_search(model, name, data, labels)
            else:
                if type(params) is dict:
                    if name == 'Support Vector Machine':
                        model = svm.SVC(kernel='rbf', probability=True, C=3.56, gamma=0.043)
                    else:
                        model = linear_model.LogisticRegression(C=2.207)
                testing_models(model, name, data, labels, t_data, t_labels)
    if log_to_file:
        sys.stdout = orig_stdout
        f.close()


if __name__ == '__main__':
    main(log_to_file=True, sparse_cv_search=False, final_run=True)
