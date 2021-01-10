import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('module://drawilleplot')
import matplotlib.pyplot as plt

from fire import Fire

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def main(data):
    pd.set_option('display.max_columns', None)
    housing = pd.read_csv(data)

    #divide median_income by 1.5 to limit income categories
    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    #capped income catogories at 5.0
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    #print(f'Income categories proportions full housing dataset:\n{income_cat_proportions(housing)}\n')
    #print(f'Stratum train set income categories proportions:\n{income_cat_proportions(strat_train_set)}\n')
    #print(f'Stratum test set income categories proportions:\n{income_cat_proportions(strat_test_set)}\n')

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) #add income_cat to test_set and train_set
    #compare_props(housing, strat_test_set, test_set)

    housing_strat_train = strat_train_set.copy()
    #housing_strat_train_plot_alpha(housing_strat_train)
    #housing_strat_train_plot_cmap(housing_strat_train)

def housing_strat_train_plot_alpha(housing_strat_train):
    housing_strat_train.plot(kind='scatter', x='longitude', y='latitude')
    housing_strat_train.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
    plt.show()

def housing_strat_train_plot_cmap(housing_strat_train):
    housing_strat_train.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                             s=housing_strat_train['population']/100, label='population', figsize=(10,7),
                             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
    plt.show()

def compare_props(housing, strat_test_set, test_set):
    compare_props = pd.DataFrame({
        'Overall': income_cat_proportions(housing),
        'Stratified': income_cat_proportions(strat_test_set),
        'Random': income_cat_proportions(test_set),
    }).sort_index()
    compare_props['Rand. %error'] = 100 * compare_props['Random']/compare_props['Overall'] - 100
    compare_props['Strat. %error'] = 100 * compare_props['Stratified']/compare_props['Overall'] - 100
    print(f'Compare stratified sampling and random sampling:\n{compare_props}')

def income_cat_proportions(data):
    return data['income_cat'].value_counts()/len(data)

if __name__ == '__main__':
    Fire(main)
