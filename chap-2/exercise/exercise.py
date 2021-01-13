import hashlib
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('module://drawilleplot')
import matplotlib.pyplot as plt

from fire import Fire
from pandas.plotting import scatter_matrix

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

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

    for set_ in(strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True) #remove income categories

    housing_strat_train = strat_train_set.copy()
    #housing_strat_train_plot_alpha(housing_strat_train)
    #housing_strat_train_plot_cmap(housing_strat_train)
    #correlations(housing_strat_train)
    #scatter_plot(housing_strat_train, 'median_income', 'median_house_value', 0.1)

    #look correlations rooms per household, bedrooms per room, and population per household
    housing_strat_train['rooms_per_household'] = housing_strat_train['total_rooms']/housing_strat_train['households']
    housing_strat_train['bedrooms_per_room'] = housing_strat_train['total_bedrooms']/housing_strat_train['total_rooms']
    housing_strat_train['population_per_household'] = housing_strat_train['population']/housing_strat_train['households']
    #correlations(housing_strat_train)
    #scatter_plot(housing_strat_train, 'rooms_per_household', 'median_house_value', 0.2)
    #print(housing_strat_train.describe()) #print descriptive statistics train set

    #prepare data for machine learning
    housing_strat_train = strat_train_set.drop('median_house_value', axis=1) #remove median house value
    housing_labels = strat_train_set['median_house_value'].copy() #use median house value as the target/label for training
    sample_incomplete_rows = housing_strat_train[housing.isnull().any(axis=1)].head()
    median = housing['total_bedrooms'].median()
    sample_incomplete_rows['total_bedrooms'].fillna(median, inplace=True)

    imputer = SimpleImputer(strategy='median')
    housing_strat_train_num = housing_strat_train.drop('ocean_proximity', axis=1) #remove text attributes because median can only calculated on numerical attributes
    imputer.fit(housing_strat_train_num)

    #check if imputer.statistics_ the same as manually computing the median of each attributes
    #print(f'imputer.statistics_:\n{imputer.statistics_}\n')
    #print(f'median of each attributes:\n{housing_strat_train_num.median().values}')

    #transform the training set
    X = imputer.transform(housing_strat_train_num)
    housing_strat_train_tr = pd.DataFrame(X, columns=housing_strat_train_num.columns, index=housing_strat_train_num.index) #tr = transform
    #print(housing_strat_train_tr.head())

    #preprocess the categorical input feature, ocean_proximity
    housing_strat_train_cat = housing_strat_train[['ocean_proximity']] #the difference between [] and [[]] is that [[]] remove Name and dtype at the bottom
    print(f'{housing_strat_train_cat.head(10)}\n')
    ordinal_encoder = OrdinalEncoder()
    housing_strat_train_cat_encoded = ordinal_encoder.fit_transform(housing_strat_train_cat)
    print(f'Ocean proximity encoded:\n{housing_strat_train_cat_encoded[:10]}\n')
    print(f'Encoder categories:\n{ordinal_encoder.categories_}')

def scatter_plot(housing, x_axis, y_axis, alpha_value):
    #attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    #scatter_matrix(housing[attributes], figsize=(12,8))
    housing.plot(kind='scatter', x=x_axis, y=y_axis, alpha=alpha_value)
    #plt.axis([0, 5, 0, 520000])
    plt.show()

def correlations(housing):
    corr_matrix = housing.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))

def housing_strat_train_plot_alpha(housing):
    #housing.plot(kind='scatter', x='longitude', y='latitude')
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
    plt.show()

def housing_strat_train_plot_cmap(housing):
    housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
                             s=housing['population']/100, label='population', figsize=(10,7),
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
