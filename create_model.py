import pandas as pd
from matplotlib import pyplot as plt
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import create_dataset as cd
import map_space as ms

n_blobs = 5
df = cd.create_nblobs(n_blobs, 5, 250)
df_space = ms.create_space(df)
for i in range(n_blobs):
    plt.plot(df.loc[df['class'] == i, 'A'], df.loc[df['class'] == i, 'B'], 'o')
plt.plot(df_space['A'], df_space['B'], 'k.', alpha=0.1)

print(len(df))
print(len(df_space))
plt.show()
cols = ['A', 'B']
df_all = df[cols].copy()
df_all['class'] = 1
df_space['class'] = 0
df_all = pd.concat([df_all, df_space], axis=0)
# model = DecisionTreeClassifier(max_depth=4)
model = SVC(C=0.001)
model.fit(df_all[cols], df_all['class'])
preds = model.predict(df_all[cols])
df_all['preds'] = preds
df_all['match'] = df_all['preds'] == df_all['class']
plt.plot(df_all.loc[(df_all['preds'] == 1) & (df_all['match']), 'A'],
         df_all.loc[(df_all['preds'] == 1) & (df_all['match']), 'B'], '.')
plt.plot(df_all.loc[(df_all['preds'] == 0) & (df_all['match']), 'A'], df_all.loc[
    (df_all['preds'] == 0) & (df_all['match']), 'B'], '.')

plt.plot(df_all.loc[(df_all['preds'] == 1) & (~df_all['match']), 'A'],
         df_all.loc[(df_all['preds'] == 1) & (~df_all['match']), 'B'], 'r.')
plt.plot(df_all.loc[(df_all['preds'] == 0) & (~df_all['match']), 'A'], df_all.loc[
    (df_all['preds'] == 0) & (~df_all['match']), 'B'], 'k.')

plt.show()
