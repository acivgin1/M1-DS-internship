import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model, svm, naive_bayes

titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']
age_intervals = [6, 18, 32, 52, 73]


def age_to_group(age):
    return float(next(x[0] for x in enumerate(age_intervals) if x[1] >= age))


def test_model(model, model_name, data, labels, t_data, t_labels):
    model.fit(data, labels)

    model_pred = model.predict(data)
    model_pred_t = model.predict(t_data)

    model_acc = np.equal(model_pred, labels).sum() / labels.shape[0]
    model_acc_t = np.equal(model_pred_t, t_labels).sum() / t_labels.shape[0]

    print('{}: training accuracy: {}, test accuracy: {}'.format(model_name, model_acc, model_acc_t))


def get_data(filename, preprocess, delete_fam_size, group_by_age):
    df1 = pd.read_csv('{}_extended_3.csv'.format(filename), na_values='')
    df2 = pd.read_csv('{}_testing_3.csv'.format(filename), na_values='')

    df1_len = df1.shape[0]
    df2_len = df2.shape[0]

    df = pd.concat([df1, df2], ignore_index=True)

    del df['surname'], df['name'], df['sex']
    if delete_fam_size:
        del df['fam_size']

    labels = df['survived'].loc[range(0, df1_len)].as_matrix()
    t_labels = df['survived'].loc[range(df1_len, df1_len + df2_len)].as_matrix()

    del df['survived']
    df['title'] = df['title'].apply(lambda x: titles_final.index(x))
    if group_by_age:
        df['age'] = df['age'].apply(age_to_group)

    df = df.as_matrix()

    if preprocess:
        df = preprocessing.scale(df)

    data = df[:df1_len]
    t_data = df[df1_len:]
    return data, labels, t_data, t_labels


def test_hypothesis():
    dectree = tree.DecisionTreeClassifier()
    svmclas = svm.SVC()
    gnbclas = naive_bayes.GaussianNB()
    logreg = linear_model.LogisticRegression(C=1e5)

    models = [dectree, svmclas, gnbclas, logreg]
    model_names = ['Decision tree', 'Support Vector Machine', 'Gaussian Naive Bayes', 'Logistic Regression']

    for delete_fam_size in [True, False]:
        for group_by_age in [False, True]:
            for preprocess in [False, True]:
                print('')
                print('Preprocess is [{}]'.format('on' if preprocess else 'off'), end=', ')
                print('Use Family Size [{}]'.format('on' if not delete_fam_size else 'off'), end=', ')
                print('Group by age is [{}]'.format('on' if group_by_age else 'off'), end='.\n')

                for model, model_name in zip(models, model_names):
                    data, labels, t_data, t_labels = get_data('Data/Titanic_dataset',
                                                              preprocess,
                                                              delete_fam_size,
                                                              group_by_age)
                    test_model(model, model_name, data, labels, t_data, t_labels)


if __name__ == '__main__':
    orig_stdout = sys.stdout
    f = open('using_median_age.txt', 'w')
    sys.stdout = f

    test_hypothesis()

    sys.stdout = orig_stdout
    f.close()
