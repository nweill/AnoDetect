import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import create_dataset as cd
import map_space as ms
from utils import pd_split_train_test


def plot_predictions(df_test):
    plt.plot(df_test.loc[(df_test['preds'] == 1) & (df_test['match']), 'A'],
             df_test.loc[(df_test['preds'] == 1) & (df_test['match']), 'B'], '.', label='in predicted in')
    plt.plot(df_test.loc[(df_test['preds'] == 0) & (df_test['match']), 'A'], df_test.loc[
        (df_test['preds'] == 0) & (df_test['match']), 'B'], '.', label='out predicted out')

    plt.plot(df_test.loc[(df_test['preds'] == 1) & (~df_test['match']), 'A'],
             df_test.loc[(df_test['preds'] == 1) & (~df_test['match']), 'B'], 'g.', label='out predited in')
    plt.plot(df_test.loc[(df_test['preds'] == 0) & (~df_test['match']), 'A'], df_test.loc[
        (df_test['preds'] == 0) & (~df_test['match']), 'B'], 'k.', label='in predicted out')
    in_p_out = df_test[(df_test['preds'] == 0) & (~df_test['match'])]
    in_p_out.apply(lambda x: plt.text(x['A'], x['B'], '%0.2f' % x['proba']), axis=1)
    plt.legend()
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
    for i in range(3, 12):
        res.append(('Decision Tree md=' + str(i), DecisionTreeClassifier(max_depth=i)))
    for i in range(-4, 3):
        c = 10 ** i
        for j in range(-4, 0):
            g = 10 ** j
            res.append(('SVC c=' + str(c) + ' g=' + str(g), SVC(C=c, gamma=g, probability=True)))
    for i in range(20, 120, 20):
        res.append(('RF n=%d' % i, RandomForestClassifier(n_estimators=i)))
    res.append(('NB', GaussianNB()))
    return res


def plot_auc_bars(results):
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


def predict_data(df, model):
    df_test = df.copy()
    cols = ['A', 'B']

    df_test['preds'] = model.predict(df_test[cols])
    df_test['match'] = df_test['preds'] == df_test['class']
    df_test['proba'] = model.predict_proba(df_test[cols])[:, 1]
    return df_test


def create_dataset(get_data_func):
    df = get_data_func()

    df_space = ms.create_space(df, cols=['A', 'B'], num_elements=len(df) * 5, margin=0.05)

    cols = ['A', 'B']
    df_all = df[cols].copy()
    df_all['class'] = 1
    df_space['class'] = 0

    df_all = pd.concat([df_all, df_space], axis=0)
    return df_all


def run_benchmark(df_test, models):
    results = pd.DataFrame()
    best_auc = -1
    best_model = None
    cols = ['A', 'B']
    for label, model in models:
        model.fit(df_train[cols], df_train['class'])
        df_test = predict_data(df_test, model)
        auc_score, fpr, tpr = get_AUC(df_test)
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
        print(label, ' ', auc_score)
        results = pd.concat([results, pd.DataFrame({'model': [label], 'auc': [auc_score]})], axis=0)
    results.reset_index(inplace=True, drop=True)
    print(results.sort_values('auc', ascending=False).head(3))
    return results, best_model


if __name__ == '__main__':
    # n_blobs = 5
    # get_data_func = cd.create_nblobs(n_blobs, 5, 250)
    get_data_func = cd.create_circular_data()
    df_all = create_dataset(get_data_func)
    df_train, df_test = pd_split_train_test(df_all, 0.3)
    models = get_list_models()
    results, best_model = run_benchmark(df_test, models)
    df_test = predict_data(df_test, best_model)
    plot_predictions(df_test)
    plot_auc_bars(results)
