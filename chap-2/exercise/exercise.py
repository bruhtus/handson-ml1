import numpy as np
import pandas as pd

from fire import Fire
from scipy.stats import expon, reciprocal

from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def main(data):
    pd.set_option('display.max_columns', None)
    housing = pd.read_csv(data)
    
    #add income categories
    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    housing_strat_train = strat_train_set.drop('median_house_value', axis=1)
    housing_strat_train_labels = strat_train_set['median_house_value'].copy()

    imputer = SimpleImputer(strategy='median')
    housing_strat_train_num = housing_strat_train.drop('ocean_proximity', axis=1)
    imputer.fit(housing_strat_train_num)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_add', FunctionTransformer(add_extra_features, validate=False, kw_args={'add_bedrooms_per_room': False, 'housing': housing_strat_train})),
        ('std_scaler', StandardScaler()),
    ])

    num_attribs = list(housing_strat_train_num)
    cat_attribs = ['ocean_proximity']

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])

    housing_strat_train_prepared = full_pipeline.fit_transform(housing_strat_train)
    #svm_gridsearchcv(housing_strat_train_prepared, housing_strat_train_labels)
    svm_randomizedsearchcv(housing_strat_train_prepared, housing_strat_train_labels)

def svm_randomizedsearchcv(housing_prepared, housing_labels):
    param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

    svm_reg = SVR()
    rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs, n_iter=50, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4, random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)

    negative_mse = rnd_search.best_score_
    rmse = np.sqrt(-negative_mse)
    with open('svm-best-score', 'w') as f:
        f.write(f'SVM Best Score: {rmse}\n')
        f.write(f'SVM Best Hyperparameters: {rnd_search.best_params_}')

def svm_gridsearchcv(housing_prepared, housing_labels):
    param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

    svm_reg = SVR()
    grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
    grid_search.fit(housing_prepared, housing_labels)
    
    negative_mse = grid_search.best_score_
    rmse = np.sqrt(-negative_mse)
    with open('svm-best-score', 'w') as f:
        f.write(f'SVM Best Score: {rmse}\n')
        f.write(f'SVM Best Hyperparameters: {grid_search.best_params_}')

def add_extra_features(X, housing, add_bedrooms_per_room=True):
    rooms_ix, bedrooms_ix, population_ix, household_ix = [list(housing.columns).index(col) for col in ('total_rooms', 'total_bedrooms', 'population', 'households')]
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]

    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

if __name__ == '__main__':
    Fire(main)
