import hashlib
import numpy as np
import pandas as pd
#import matplotlib
#matplotlib.use('module://drawilleplot')
#import matplotlib.pyplot as plt

from fire import Fire

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer #to replace missing values
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

#from sklearn.pipeline import FeatureUnion
#from pandas.plotting import scatter_matrix
#from sklearn.preprocessing import LabelEncoder
#from sklearn.preprocessing import LabelBinarizer
#from sklearn.preprocessing import Imputer #deprecated

def main(data):
    pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    housing = pd.read_csv(data)
    housing_with_id = housing.reset_index() #add an 'index' column
    housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
    #train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    #print(f"Income category:\n{housing['income_cat'].value_counts()/len(housing)}\n")
    #print(f'test_set head:\n{test_set.head()}\n')

    ##display histogram
    #housing.hist(bins=50, figsize=(15,15))
    #housing['median_income'].hist()
    #housing['income_cat'].hist()
    #print(f'{plt.show()}\n')

    ##show income category train and test set
    #print(f"income category test set:\n{strat_test_set['income_cat'].value_counts()/len(strat_test_set)}\n")
    #print(f"income category train set:\n{strat_train_set['income_cat'].value_counts()/len(strat_test_set)}\n")

    ##compare properties
    #train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) #add income_cat to test_set and train_set
    #compare_props = pd.DataFrame({
    #    'Overall': income_cat_proportions(housing),
    #    'Stratified': income_cat_proportions(strat_test_set),
    #    'Random': income_cat_proportions(test_set),
    #}).sort_index()
    #compare_props['Rand. %error'] = 100 * compare_props['Random']/compare_props['Overall'] - 100
    #compare_props['Strat. %error'] = 100 * compare_props['Stratified']/compare_props['Overall'] - 100
    #print(f'compare stratified sampling and random sampling:\n{compare_props}')

    ##remove income_cat attribute
    #for set in (strat_train_set, strat_test_set):
    #    set.drop(['income_cat'], axis=1, inplace=True)

    #housing = strat_train_set.copy() #create a copy of train set
    ##plotting again
    #housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
    #             s=housing['population']/100, label='population',
    #             c='median_house_value', cmap=plt.get_cmap('jet'),
    #             colorbar=True,)
    #plt.show()

    ##looking for correlations
    #corr_matrix = housing.corr()
    #print(corr_matrix['median_house_value'].sort_values(ascending=False))

    ##anoher way looking for correlations
    #attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    #scatter_matrix(housing[attributes], figsize=(12, 8))
    #housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    #plt.show()

    #look at rooms per households and bedrooms per rooms
    #housing['rooms_per_households'] = housing['total_rooms']/housing['households']
    #housing['bedrooms_per_rooms'] = housing['total_bedrooms']/housing['total_rooms']
    #housing['population_per_household'] = housing['population']/housing['households']
    #corr_matrix = housing.corr()
    #print(corr_matrix['median_house_value'].sort_values(ascending=False))

    housing = strat_train_set.drop('median_house_value', axis=1) #doesn't effect strat_train_set
    housing_labels = strat_train_set['median_house_value'].copy()

    imputer = SimpleImputer(strategy='median')
    housing_num = housing.drop('ocean_proximity', axis=1)

    imputer.fit(housing_num) #fit the imputer instance to training data
    #print(imputer.statistics_)
    #print(housing_num.median().values) #using imputer values

    #x = imputer.transform(housing_num) #numpy array
    #housing_tr = pd.DataFrame(x, columns=housing_num.columns) #put numpy array to dataframe

    #encoder = LabelEncoder()
    #housing_cat = housing['ocean_proximity']
    #housing_cat_encoded = encoder.fit_transform(housing_cat)
    #print(housing_cat_encoded)
    #print(encoder.classes_) #IH OCEAN is mapped to 0, INLAND is mapped to 1, and so on
    #onehot = OneHotEncoder()
    #housing_cat_OH = onehot.fit_transform(housing_cat_encoded.reshape(-1,1))
    #print(housing_cat_OH)

    #encoder = LabelBinarizer() #from text categories to integer categories then from integer categories to one-hot vectors (default numpy array)
    #encoder = LabelBinarizer(sparse_output=True) #from text categories to integer categories then from integer categories to one-hot vectors (sparse matrix instead)
    #housing_cat_OH = encoder.fit_transform(housing_cat)
    #print(housing_cat_OH)

    #attr_adder_caa = CombinedAttributesAdder(add_bedrooms_per_room=False) #CombinedAttributesAdder
    #housing_extra_attribs_caa = attr_adder_caa.transform(housing.values)
    #print(f'default:\n{housing_extra_attribs_caa}\n')
    #attr_adder_ft = FunctionTransformer(add_extra_features, validate=False, kw_args={"add_bedrooms_per_room":False}) #FunctionTransformer
    #housing_extra_attribs_ft = attr_adder_ft.fit_transform(housing.values)
    #print(f'FunctionTransformer:\n{housing_extra_attribs_ft}')

    #housing_extra_attribs = pd.DataFrame(
    #    housing_extra_attribs_ft,
    #    columns=list(housing.columns)+["rooms_per_household", "populations_per_household"],
    #    index=housing.index)
    #print(housing_extra_attribs.head())

    #build a pipeline

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

    #housing_num_tr = num_pipeline.fit_transform(housing_num)
    #print(housing_num_tr)

    num_attribs = list(housing_num)
    cat_attribs = ['ocean_proximity']

    #old_num_pipeline = Pipeline([
    #    ('selector', OldDataFrameSelector(num_attribs)),
    #    ('imputer', SimpleImputer(strategy='median')),
    #    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    #    ('std_scaler', StandardScaler()),
    #])

    #old_cat_pipeline = Pipeline([
    #    ('selector', OldDataFrameSelector(cat_attribs)),
    #    ('cat_encoder', OneHotEncoder(sparse=False)),
    #])

    #old_full_pipeline = FeatureUnion([
    #    ('num_pipeline', old_num_pipeline),
    #    ('cat_pipeline', old_cat_pipeline),
    #])

    #old_housing_prepared = old_full_pipeline.fit_transform(housing)
    #print(f'FeatureUnion pipeline:\n{old_housing_prepared}\n')

    new_full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])

    new_housing_prepared = new_full_pipeline.fit_transform(housing)
    #print(f'ColumnTransformer Pipeline:\n{new_housing_prepared}\n')

    #random_forest(new_housing_prepared, housing_labels)
    #decision_tree(new_housing_prepared, housing_labels)
    #linear_regression(new_housing_prepared, housing_labels, housing)
    #svm(new_housing_prepared, housing_labels)

def random_forest(pipeline, housing_labels):
    forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
    forest_reg.fit(pipeline, housing_labels)

    housing_predictions_forest_reg = forest_reg.predict(pipeline)
    forest_mse = mean_squared_error(housing_labels, housing_predictions_forest_reg)
    forest_rmse = np.sqrt(forest_mse)
    #print(forest_rmse)

    forest_scores = cross_val_score(forest_reg, pipeline, housing_labels, scoring='neg_mean_squared_error', cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)

    print('random forest:')
    print(f'Scores:\n{forest_rmse_scores}')
    print(f'Mean:{forest_rmse_scores.mean()}')
    print(f'Standard deviation:{forest_rmse_scores.std()}\n')

def decision_tree(pipeline, housing_labels):
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(pipeline, housing_labels)

    housing_predictions_tree_reg = tree_reg.predict(pipeline)
    tree_mse = mean_squared_error(housing_labels, housing_predictions_tree_reg)
    tree_mse = np.sqrt(tree_mse)
    #print(f'Mean Squared Error(Decision Tree): {tree_mse}\n')

    tree_scores = cross_val_score(tree_reg, pipeline, housing_labels, scoring='neg_mean_squared_error', cv=10)
    tree_rmse_scores = np.sqrt(-tree_scores)

    print('decision tree:')
    print(f'Scores:\n{tree_rmse_scores}')
    print(f'Mean: {tree_rmse_scores.mean()}')
    print(f'Standard deviation: {tree_rmse_scores.std()}\n')

def linear_regression(pipeline, housing_labels, housing):
    lin_reg = LinearRegression()
    lin_reg.fit(pipeline, housing_labels)

    #some_data = housing.iloc[:5]
    #some_labels = housing_labels.iloc[:5]
    #some_data_prepared = new_full_pipeline.transform(some_data)
    #print(f'Prediction:\n{lin_reg.predict(some_data_prepared)}\n')
    #print(f'Labels:\n{list(some_labels)}\n')

    housing_predictions_lin_reg = lin_reg.predict(pipeline)
    lin_mse = mean_squared_error(housing_labels, housing_predictions_lin_reg)
    lin_mse = np.sqrt(lin_mse)
    print(f'Mean Squared Error(linear regression): {lin_mse}\n')

    lin_scores = cross_val_score(lin_reg, pipeline, housing_labels, scoring='neg_mean_squared_error', cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)

    print('linear regression:')
    print(f'Scores:\n{lin_rmse_scores}')
    print(f'Mean: {lin_rmse_scores.mean()}')
    print(f'Standard deviation: {lin_rmse_scores.std()}\n')

def svm(pipeline, housing_labels):
    svm_reg = SVR(kernel='linear')
    svm_reg.fit(pipeline, housing_labels)

    housing_predictions_svm_reg = svm_reg.predict(pipeline)
    svm_mse = mean_squared_error(housing_labels, housing_predictions_svm_reg)
    svm_rmse = np.sqrt(svm_mse)
    print(svm_rmse)

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

def income_cat_proportions(data):
    return data['income_cat'].value_counts()/len(data)

def add_extra_features(X, add_bedrooms_per_room=True):
    rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
    population_per_household = X[:, population_ix] / X[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[X, rooms_per_household, population_per_household]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix]/X[:, household_ix]
        population_per_household = X[:, population_ix]/X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values

if __name__ == '__main__':
    Fire(main)