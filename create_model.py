import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('ggplot')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import create_dataset as cd
import map_space as ms
from utils import pd_split_train_test
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


def _get_data_classified_per_match(df_test):
    in_pred_in = df_test[(df_test['preds'] == 1) & (df_test['match'])]
    out_pred_out = df_test[(df_test['preds'] == 0) & (df_test['match'])]
    in_pred_out = df_test[(df_test['preds'] == 0) & (~df_test['match'])]
    out_pred_in = df_test[(df_test['preds'] == 1) & (~df_test['match'])]
    return in_pred_in, out_pred_out, in_pred_out, out_pred_in


def plot_predictions(df_test, show_proba=False):
    in_pred_in, out_pred_out, in_pred_out, out_pred_in = _get_data_classified_per_match(df_test)
    plt.plot(in_pred_in['A'], in_pred_in['B'], '.', label='in predicted in')
    plt.plot(out_pred_out['A'], out_pred_out['B'], '.', label='out predicted out')

    plt.plot(in_pred_out['A'], in_pred_out['B'], 'g.', label='in predicted out')
    plt.plot(out_pred_in['A'], out_pred_in['B'], 'k.', label='out predited in')
    if show_proba:
        in_pred_out.apply(lambda x: plt.text(x['A'], x['B'], '%0.2f' % x['proba']), axis=1)
    plt.legend()
    plt.show()


def get_conf_matrix(df_test):
    in_pred_in, out_pred_out, in_pred_out, out_pred_in = _get_data_classified_per_match(df_test)
    TP = len(in_pred_in)
    TN = len(out_pred_out)
    FP = len(out_pred_in)
    FN = len(in_pred_out)
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'Precision': TP / (TP + FP), 'Recall': TP / (TP + FN)}


def plot_data(df_all):
    df = df_all[df_all['class'] == 1]
    df_space = df_all[df_all['class'] == 0]
    n_cat = len(df['cat'].unique())
    for i in range(n_cat):
        plt.plot(df.loc[df['cat'] == i, 'A'], df.loc[df['cat'] == i, 'B'], 'o')
    plt.plot(df_space['A'], df_space['B'], 'k.', alpha=0.1)
    plt.show()


def get_AUC(df_test, out_col='class'):
    fpr, tpr, thresholds = roc_curve(df_test[out_col], df_test['proba'])
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr


def plot_auc(fpr, tpr, auc_score, label=''):
    plt.plot(fpr, tpr,
             lw=2, label=label + ' (area = %0.2f)' % auc_score)


def get_list_models():
    res = []
    res.append(('logistic Regression', LogisticRegression()))
    res.append(('NB', GaussianNB()))

    for i in range(3, 12):
        res.append(('Decision Tree md=' + str(i), DecisionTreeClassifier(max_depth=i)))
    for i in range(2, 4):
        c = 10 ** i
        for j in range(-1, 2):
            g = 10 ** j
            res.append(('SVC c=' + str(c) + ' g=' + str(g), SVC(C=c, gamma=g, probability=True)))
    # for i in range(20, 200, 20):
    #     res.append(('RF n=%d' % i, RandomForestClassifier(n_estimators=i)))
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


def create_dataset(get_data_func, density=2):
    df = get_data_func()

    df_space = ms.create_space(df, cols=['A', 'B'], num_elements=len(df) * density, margin=0.05)

    cols = ['A', 'B', 'cat']
    df_all = df[cols].copy()
    df_all['class'] = 1
    df_space['class'] = 0

    df_all = pd.concat([df_all, df_space], axis=0, sort=True)
    return df_all


def run_benchmark(df_train, df_test, models, cols=['A', 'B'], output_col='class'):
    results = pd.DataFrame()
    best_auc = -1
    best_model = None
    for label, model in models:
        model.fit(df_train[cols], df_train[output_col])
        df_test = predict_data(df_test, model)
        auc_score, fpr, tpr = get_AUC(df_test, output_col)
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
        print(label, ' ', auc_score)
        results = pd.concat([results, pd.DataFrame({'model': [label], 'auc': [auc_score]})], axis=0)
    results.reset_index(inplace=True, drop=True)
    print(results.sort_values('auc', ascending=False).head(3))
    return results, best_model


if __name__ == '__main__':
    # get_data_func = cd.create_nblobs(8, 5, 250)
    get_data_func = cd.create_circular_data(radius=[1, 3.5, 4, 15, 6, 7, 18, 9], n_per_class=1000)
    df_all = create_dataset(get_data_func, density=2)
    plot_data(df_all)
    df_train, df_test = pd_split_train_test(df_all, 0.5)
    models = get_list_models()
    results, best_model = run_benchmark(df_train, df_test, models)
    df_test = predict_data(df_test, best_model)
    df_train = predict_data(df_train, best_model)
    print(get_conf_matrix(df_train))

    print(get_conf_matrix(df_test))

    plot_predictions(df_test)
    plot_predictions(df_train)

    plot_auc_bars(results)
    model = DecisionTreeClassifier()
    train_init = df_train[df_train['class'] == 1]
    test_init = df_test[df_test['class'] == 1]
    model.fit(train_init[['A', 'B']], train_init['cat'])
    test_init['preds_cat'] = model.predict(test_init[['A', 'B']])
    print(confusion_matrix(test_init['preds_cat'], test_init['cat']))
    df_test_cleaned = test_init[test_init['preds'] == 1]
    print(confusion_matrix(df_test_cleaned['preds_cat'], df_test_cleaned['cat']))
