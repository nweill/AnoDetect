import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import create_dataset as cd
import map_space as ms
from utils import pd_split_train_test


def plot_predictions(df_test):
    plt.plot(df_test.loc[(df_test['preds'] == 1) & (df_test['match']), 'A'],
             df_test.loc[(df_test['preds'] == 1) & (df_test['match']), 'B'], '.')
    plt.plot(df_test.loc[(df_test['preds'] == 0) & (df_test['match']), 'A'], df_test.loc[
        (df_test['preds'] == 0) & (df_test['match']), 'B'], '.')

    plt.plot(df_test.loc[(df_test['preds'] == 1) & (~df_test['match']), 'A'],
             df_test.loc[(df_test['preds'] == 1) & (~df_test['match']), 'B'], 'r.')
    plt.plot(df_test.loc[(df_test['preds'] == 0) & (~df_test['match']), 'A'], df_test.loc[
        (df_test['preds'] == 0) & (~df_test['match']), 'B'], 'k.')

    plt.show()


def plot_data(df, n_blobs):
    for i in range(n_blobs):
        plt.plot(df.loc[df['class'] == i, 'A'], df.loc[df['class'] == i, 'B'], 'o')
    plt.plot(df_space['A'], df_space['B'], 'k.', alpha=0.1)
    plt.show()


def get_AUC(df_test):
    fpr, tpr, thresholds = roc_curve(df_test['class'], df_test['proba'])
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr


def plot_auc(fpr, tpr, auc_score, label=''):
    plt.plot(fpr, tpr,
             lw=2, label=label + ' (area = %0.2f)' % auc_score)


def get_list_models():
    res = []
    res.append(('logistic Regression', LogisticRegression()))
    for i in range(7, 10):
        res.append(('Decision Tree md=' + str(i), DecisionTreeClassifier(max_depth=i)))
    for i in range(-2, 2):
        c = 10 ** i
        for j in range(-2, 2):
            g = 10 ** j
            res.append(('SVC c=' + str(c) + ' g=' + str(g), SVC(C=c, gamma=g, probability=True)))

    return res


def plot_AUCs(results):
    ax = range(len(results))
    colors = ['blue'] * len(results)
    best = results[results['auc'] == results['auc'].max()].index[0]
    print(best)
    colors[best] = 'red'
    plt.bar(ax, results['auc'], color=colors)
    plt.xticks(ax, results['model'], rotation='vertical')
    plt.ylim([0.05, 1.05])
    for i in range(len(results)):
        plt.text(ax[i], list(results['auc'].apply(lambda x: x + 0.02))[i],
                 list(results['auc'].apply(lambda x: '%0.3f' % x))[i], horizontalalignment='center')
    plt.tight_layout()
    plt.show()


# create datasets
n_blobs = 5
df = cd.create_nblobs(n_blobs, 5, 250)

df_space = ms.create_space(df)

cols = ['A', 'B']
df_all = df[cols].copy()
df_all['class'] = 1
df_space['class'] = 0

df_all = pd.concat([df_all, df_space], axis=0)

df_train, df_test = pd_split_train_test(df_all)
models = get_list_models()
results = pd.DataFrame()
plot_curve = False
for label, model in models:
    model.fit(df_train[cols], df_train['class'])
    df_test['preds'] = model.predict(df_test[cols])
    df_test['match'] = df_test['preds'] == df_test['class']
    print(label)
    df_test['proba'] = model.predict_proba(df_test[cols])[:, 1]
    auc_score, fpr, tpr = get_AUC(df_test)
    print(label, ' ', auc_score)
    if plot_curve:
        plot_auc(fpr, tpr, auc_score, label)
    results = pd.concat([results, pd.DataFrame({'model': [label], 'auc': [auc_score]})], axis=0)
results.reset_index(inplace=True, drop=True)
plot_AUCs(results)
if plot_curve:
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
