import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

titles_final = ['Mr', 'Mrs', 'Miss', 'Ms', 'Master',
                'Madame', 'Mlle',
                'Captain', 'Colonel', 'Col', 'Major',
                'Sir', 'Lady',
                'Dr',
                'Rev']

df = pd.read_csv('Data/Titanic_dataset_extended_2.csv', na_values='')
df['died'] = np.ones((df.shape[0], 1), dtype=np.uint8) - df['survived'].as_matrix().reshape((-1, 1))
df['ones'] = np.ones((df.shape[0], 1), dtype=np.uint8)

age_intervals = [0, 1, 3, 6, 12, 18, 22, 27, 32, 37, 42, 47, 52, 57, 62, 72]
df['age_interval'] = pd.cut(df.age, age_intervals)

# survive_title = df.groupby(['title']).agg({'survived': ['sum'], 'died': ['sum']})
# survive_sex = df.groupby(['sex']).agg({'survived': ['sum'], 'died': ['sum']})
# survive_class = df.groupby(['p_class']).agg({'survived': ['sum'], 'died': ['sum']})
# survive_age_sex = df.groupby(['age_interval', 'sex']).agg({'survived': ['sum'], 'died': ['sum']})
# survive_age_sex_class = df.groupby(['age_interval', 'sex', 'p_class']).agg({'survived': ['sum'], 'died': ['sum']})
# survive_age_title = df.groupby(['age_interval', 'title']).agg({'survived': ['sum'], 'died': ['sum']})
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

surv1 = survive_age_title_class_famsize['survived']['sum'].as_matrix().reshape((-1, 1))
died1 = survive_age_title_class_famsize['died']['sum'].as_matrix().reshape((-1, 1))
total1 = survive_age_title_class_famsize['ones']['sum'].as_matrix().reshape((-1, 1))
accuracy1 = np.maximum(surv1, died1).sum() / total1.sum()
print(accuracy1)

survive_age_title_class_famsize.to_csv('Data/survive_age_title_class_famsize.csv')

# print(survive_title)
# print(survive_sex)
# print(survive_class)
# print(survive_age_sex)
# print(survive_age_sex_class)
# print(survive_age_title)
# print(survive_age_title_class)

del df['surname'], df['name'], df['sex'], df['died'], df['ones'], df['age_interval']
labels = df['survived'].as_matrix()
del df['survived']

df['title'] = df['title'].apply(lambda x: titles_final.index(x))
features = df.as_matrix()

# # plot age title
# fig = plt.figure()
# plt.xlabel('Age')
# plt.ylabel('Title')
# plt.yticks(range(0, len(titles_final), 1), titles_final)
#
# for elem, survived in zip(features, labels):
#     if survived:
#         plt.plot(elem[1], elem[0], 'ro')
#     else:
#         plt.plot(elem[1], elem[0], 'bv')
#
# fig.savefig('Data/Images/Age_Title.png', bbox_inches='tight')
#
# # plot age title class
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.set_xlabel('Passenger class')
# ax.set_xticks([3, 2, 1])
#
# ax.set_ylabel('Age')
#
# ax.set_zlabel('Title')
# ax.set_zticks(np.arange(0, len(titles_final), 1))
# ax.set_zticklabels(titles_final)
#
# for elem, survived in zip(features, labels):
#     if survived:
#         ax.scatter(elem[2], elem[1], elem[0], color='r', marker='o')
#     else:
#         ax.scatter(elem[2], elem[1], elem[0], color='b', marker='v')
#
# fig.savefig('Data/Images/Scatter.png', bbox_inches='tight')
