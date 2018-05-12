import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

titles_final = ['Dr', 'Rev', 'Col', 'Mr', 'Sir', 'Master', 'Lady', 'Mrs', 'Ms']

df = pd.read_csv('Data/Titanic_dataset_extended_1.csv', na_values='')
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


survived = survive_age_title_class['survived']['sum'].as_matrix().reshape((-1, 1))
died = survive_age_title_class['died']['sum'].as_matrix().reshape((-1, 1))
total = survive_age_title_class['ones']['sum'].as_matrix().reshape((-1, 1))

accuracy = np.maximum(survived, died).sum() / total.sum()
print(accuracy)

survive_age_title_class.to_csv('Data/age_title_class.csv')

print(survive_title)
print(survive_sex)
print(survive_class)
print(survive_age_sex)
print(survive_age_sex_class)
print(survive_age_title)
print(survive_age_title_class)

del df['surname'], df['name'], df['sex'], df['died'], df['ones'], df['age_interval']
labels = df['survived'].as_matrix()
del df['survived']

df['title'] = df['title'].apply(lambda x: titles_final.index(x))
features = df.as_matrix()

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for f, l in zip(features, labels):
    if l:
        ax.scatter(f[2], f[1], f[0], color='r', marker='o')
    else:
        ax.scatter(f[2], f[1], f[0], color='b', marker='x')

ax.set_xlabel('PClass')
ax.set_xticks([1, 2, 3])
ax.set_ylabel('Age Group')
ax.set_zlabel('Title')
ax.set_zticks(np.arange(0, len(titles_final), 1))
ax.set_zticklabels(titles_final)

plt.show(block=True)
