import numpy as np
import pandas as pd


titles_final = ['Mr', 'Mrs', 'Ms', 'Master', 'Col', 'Sir', 'Lady', 'Dr', 'Rev']

df = pd.read_csv('Data/Titanic_dataset_extended.csv', na_values=[100])
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
