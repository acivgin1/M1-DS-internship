import pandas as pd
from sklearn import preprocessing
from sklearn import tree, linear_model

titles_final = ['Mr', 'Mrs', 'Miss', 'Ms', 'Master',
                'Madame', 'Mlle',
                'Captain', 'Colonel', 'Col', 'Major',
                'Sir', 'Lady',
                'Dr',
                'Rev']

df1 = pd.read_csv('Data/Titanic_dataset_extended_2.csv', na_values='')
df2 = pd.read_csv('Data/Titanic_dataset_testing_2.csv', na_values='')

df1_len = df1.shape[0]
df2_len = df2.shape[0]

df = pd.concat([df1, df2], ignore_index=True)

del df['surname'], df['name'], df['sex'], df['fam_size']
labels = df['survived'].loc[range(0, df1_len)].as_matrix()
t_labels = df['survived'].loc[range(df1_len, df1_len + df2_len)].as_matrix()

del df['survived']

df['title'] = df['title'].apply(lambda x: titles_final.index(x))

df_matrix = preprocessing.scale(df.as_matrix())

features = df[:df1_len]
t_features = df[df1_len:]

clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
prediction = clf.predict(t_features)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(features, labels)
logreg_pred = logreg.predict(t_features)