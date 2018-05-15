import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model, svm, naive_bayes
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']
age_intervals = [6, 18, 32, 52, 72, 100]


def age_to_group(age):
    return float(next(x[0]+1 for x in enumerate(age_intervals) if x[1] >= age))


def test_model(model, model_name, data, labels, t_data, t_labels, use_cross_validation=False, final=False):
    if use_cross_validation:
        model.fit(data, labels)
        print("Best parameters set found on development set:")
        if not model_name == 'Gaussian Naive Bayes':
            print(model.best_params_)
            print('Best score is: {}'.format(model.best_score_))
    else:
        model.fit(data, labels)

        model_pred = model.predict(data)
        model_pred_t = model.predict(t_data)

        model_acc = np.equal(model_pred, labels).sum() / labels.shape[0]
        model_acc_t = np.equal(model_pred_t, t_labels).sum() / t_labels.shape[0]

        model_precision = average_precision_score(t_labels, model_pred_t)
        f1_score = precision_recall_fscore_support(t_labels, model_pred_t, average='micro')

        print('{}: acc: {}, t_acc: {}, prec: {}, f1: {}'.format(model_name, model_acc, model_acc_t, model_precision, f1_score))

        if final:
            model_pred_t_prob = model.predict_proba(t_data)
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

    svm_parameters = {'C': [0.001, 0.01, 0.1, 1, 10],
                      'gamma': [0.001, 0.01, 0.1, 1]}
    svmclas = svm.SVC(kernel='rbf')

    gnbclas = naive_bayes.GaussianNB()

    logreg_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    logreg = linear_model.LogisticRegression(C=1e5)

    models = [dectree, svmclas, gnbclas, logreg]
    model_names = ['Decision tree', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Logistic Regression']
    parameters = [dec_parameters, svm_parameters, {}, logreg_parameters]

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
                        print('\n{}'.format(model_name))
                        model = GridSearchCV(model,
                                             param_grid=model_parameters,
                                             cv=5,
                                             scoring='f1')

                    data, labels, t_data, t_labels = get_data(filename,
                                                              preprocess,
                                                              delete_fam_size,
                                                              group_by_age,
                                                              training)

                    test_model(model, model_name, data, labels, t_data, t_labels, use_cross_validation)


def main(log_to_file):
    if log_to_file:
        f = open('gridSearchCV.txt', 'w')
        f.close()
    i = 1
    for mean in [False, True]:
        for std_dev_scale in [0, 0.2, 0.4, 0.6, 0.8]:
            for better_title in [False, True]:
                if log_to_file:
                    orig_stdout = sys.stdout
                    f = open('gridSearchCV.txt', 'a')
                    sys.stdout = f

                filename = 'Data/Datasets/Titanic_{}'.format(i)
                print('\n############################################')
                print('mean[{}], std_dev_scale[{}], better_title[{}]\n'.format(mean, std_dev_scale, better_title))
                test_hypothesis(use_cross_validation=True, filename=filename, training=True)
                i += 1

                if log_to_file:
                    sys.stdout = orig_stdout
                    f.close()


if __name__ == '__main__':
    main(log_to_file=True)
