import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools


def plot_confusion_matrix(cm, xticks,
                          yticks,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float')

        #cm = cm / cm.sum(axis=1)[:, np.newaxis]
        cm = cm / cm.sum(axis=0)[np.newaxis, :]

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xtick_marks = np.arange(len(xticks))
    ytick_marks = np.arange(len(yticks))
    plt.xticks(xtick_marks, xticks, rotation=45)
    plt.yticks(ytick_marks, yticks)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']

df = pd.read_csv('Data/Titanic_dataset_extended.csv', na_values=[''])
df['title'].replace(to_replace=titles_final, value=range(9), inplace=True)
df['title'].astype(int)

plt.figure()

df_survived = df.loc[df.loc[:, 'survived'] == 1, :]
df_dead = pd.DataFrame(df.loc[df.loc[:, 'survived'] == 0, :])
df_dead['survived'].replace(to_replace=0, value=1, inplace=True)

table_survivors = pd.pivot_table(df_survived, values='survived', index=['title'], columns=['age'], aggfunc=np.sum)
table_dead = pd.pivot_table(df_dead, values='survived', index=['title'], columns=['age'], aggfunc=np.sum)

table_survivors.fillna(0, inplace=True)
table_dead.fillna(0, inplace=True)

plot_confusion_matrix(table_survivors.as_matrix().astype(np.uint16),
                      xticks=table_survivors.columns.values,
                      yticks=table_survivors.index.values,
                      normalize=True)
plt.figure()
plot_confusion_matrix(table_dead.as_matrix().astype(np.uint16),
                      xticks=table_dead.columns.values,
                      yticks=table_dead.index.values,
                      normalize=True)

plt.show()

plt.figure()
plot_confusion_matrix(table_dead.as_matrix().astype(np.uint16),
                      xticks=table_dead.columns.values,
                      yticks=table_dead.index.values,
                      normalize=False)
plt.show()
