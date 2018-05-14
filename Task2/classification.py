import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model, svm, naive_bayes
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.model_selection import GridSearchCV, cross_validate

# TODO: implement cross validation, and precision/recall/f1_score

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']
age_intervals = [6, 18, 32, 52, 72]


def age_to_group(age):
    return float(next(x[0] for x in enumerate(age_intervals) if x[1] >= age))


def test_model(model, model_name, data, labels, t_data, t_labels, use_cross_validation=False, final=False):

    if use_cross_validation:
        model.fit(data, labels)
        print("Best parameters set found on development set:")
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


def get_data(filename, preprocess, delete_fam_size, group_by_age):
    df1 = pd.read_csv('{}_extended_4.csv'.format(filename), na_values='')
    df2 = pd.read_csv('{}_testing_4.csv'.format(filename), na_values='')

    # df = df.sample(frac=1).reset_index(drop=True)
    # df_len = df.shape[0]

    # df1 = df.iloc[:df_len-400, :]
    # df2 = df.iloc[df_len-400:, :]

    df1_len = df1.shape[0]
    df2_len = df2.shape[0]
    # df3_len = df3.shape[0]

    df = pd.concat([df1, df2], ignore_index=True)

    del df['surname'], df['name'], df['sex']
    if delete_fam_size:
        del df['fam_size']

    labels = df['survived'].loc[range(0, df1_len)].as_matrix()
    t_labels = df['survived'].loc[range(df1_len, df1_len + df2_len)].as_matrix()

    del df['survived']
    df['title'] = df['title'].apply(lambda x: titles_final.index(x)+1)
    if group_by_age:
        df['age'] = df['age'].apply(age_to_group)

    df = df.as_matrix()

    if preprocess:
        df = preprocessing.scale(df)

    data = df[:df1_len]
    t_data = df[df1_len:]
    return data, labels, t_data, t_labels


def test_hypothesis(use_cross_validation):
    dectree = tree.DecisionTreeClassifier()

    svm_parameters = {'kernel': ('linear', 'rbf'), 'C': np.arange(0.1, 20, 0.5)}
    svmclas = svm.SVC(probability=True)

    gnbclas = naive_bayes.GaussianNB()

    logreg = linear_model.LogisticRegression(C=1e5)

    models = [dectree, svmclas, gnbclas, logreg]
    model_names = ['Decision tree', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Logistic Regression']
    parameters = [None, svm_parameters, None, None]

    for delete_fam_size in [True, False]:
        for group_by_age in [False, True]:
            for preprocess in [False, True]:
                print('')
                print('scale[{}]'.format('on' if preprocess else 'off'), end=', ')
                print('fam_size[{}]'.format('on' if not delete_fam_size else 'off'), end=', ')
                print('group_age[{}]'.format('on' if group_by_age else 'off'), end='.\n')

                for model, model_name, model_parameters in zip(models, model_names, parameters):
                    if use_cross_validation:
                        model = GridSearchCV(model,
                                             param_grid=model_parameters,
                                             cv=10,
                                             return_train_score=True,
                                             scoring='f1')

                    data, labels, t_data, t_labels = get_data('Data/Titanic_dataset',
                                                              preprocess,
                                                              delete_fam_size,
                                                              group_by_age)

                    test_model(model, model_name, data, labels, t_data, t_labels, use_cross_validation)


if __name__ == '__main__':
    # orig_stdout = sys.stdout
    # f = open('final_results.txt', 'w')
    # sys.stdout = f
    for mean in [False, True]:
        for std_dev_scale in [0, 0.2, 0.4, 0.6, 0.8]:
            for better_title in [False, True]:
                test_hypothesis()

    # sys.stdout = orig_stdout
    # f.close()
