import os
import numpy as np

from fire import Fire
from sklearn.datasets import fetch_openml

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def main():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)

    sort_by_target(mnist)

    X, y = mnist['data'], mnist['target']

    digit = y[36000]
    some_digit = X[36000]

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    knn_clf = KNeighborsClassifier()
    sgd_clf = SGDClassifier(max_iter=5, tol=np.infty, random_state=42)
    forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

    # knn_gridsearch(knn_clf, X_train, y_train, X_test, y_test)
    # sgd_gridsearch(sgd_clf, X_train, y_train, X_test, y_test)
    # forest_gridsearch(forest_clf, X_train, y_train, X_test, y_test)

    # knn_randomizedsearch(knn_clf, X_train, y_train, X_test, y_test)
    sgd_randomizedsearch(sgd_clf, X_train, y_train, X_test, y_test)
    forest_randomizedsearch(forest_clf, X_train, y_train, X_test, y_test)


def sort_by_target(mnist):
    reorder_train = np.array(
        sorted([(target, i) for i, target in enumerate(mnist.target[:60000])])
    )[:, 1]
    reorder_test = np.array(
        sorted([(target, i) for i, target in enumerate(mnist.target[60000:])])
    )[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def knn_gridsearch(knn, X_train, y_train, X_test, y_test):
    knn.fit(X_train, y_train)
    param_grid = [{
        'weights': ['uniform', 'distance'],
        'n_neighbors': [3, 4, 5]
        }]
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    accuracy = accuracy_score(y_test, y_pred)
    with open('knn-grid-search', 'w') as f:
        f.write(f'KNN Best Parameter: {best_params}\n')
        f.write(f'KNN Best Score: {best_score}')

    with open('knn-accuracy-score', 'w') as f:
        f.write(f'KNN Accuracy Score: {accuracy}')


def sgd_gridsearch(sgd, X_train, y_train, X_test, y_test):
    sgd.fit(X_train, y_train)
    param_grid = [{
        'loss': ['hinge', 'log', 'modified_huber'],
        'alpha': [0.0001, 0.00001, 0.000001]
        }]
    grid_search = GridSearchCV(sgd, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    accuracy = accuracy_score(y_test, y_pred)
    with open('sgd-grid-search', 'w') as f:
        f.write(f'SGD Best Parameter: {best_params}\n')
        f.write(f'SGD Best Score: {best_score}')

    with open('sgd-accuracy-score', 'w') as f:
        f.write(f'SGD Accuracy Score: {accuracy}')


def forest_gridsearch(forest, X_train, y_train, X_test, y_test):
    forest.fit(X_train, y_train)
    param_grid = [{
        'n_estimators': [3, 10, 30],
        'criterion': ['gini', 'entropy'],
        "max_features": [2, 4, 6, 8],
        }]
    grid_search = GridSearchCV(forest, param_grid, cv=5, verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    accuracy = accuracy_score(y_test, y_pred)
    with open('forest-grid-search', 'w') as f:
        f.write(f'Random Forest Best Parameter: {best_params}\n')
        f.write(f'Random Forest Best Score: {best_score}\n')

    with open('forest-accuracy-score', 'w') as f:
        f.write(f'Random Forest Accuracy Score: {accuracy}')


def knn_randomizedsearch(knn, X_train, y_train, X_test, y_test):
    knn.fit(X_train, y_train)
    param_distribs = [{
        'weights': ['uniform', 'distance'],
        'n_neighbors': [3, 4, 5]
        }]
    randomized_search = RandomizedSearchCV(
            knn,
            param_distribs,
            cv=5,
            verbose=3,
            n_jobs=-1
            )
    randomized_search.fit(X_train, y_train)
    y_pred = randomized_search.predict(X_test)
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_
    accuracy = accuracy_score(y_test, y_pred)
    with open('knn-randomized-search', 'w') as f:
        f.write(f'KNN Best Parameter: {best_params}\n')
        f.write(f'KNN Best Score: {best_score}')

    with open('knn-accuracy-score', 'w') as f:
        f.write(f'KNN Accuracy Score: {accuracy}')


def sgd_randomizedsearch(sgd, X_train, y_train, X_test, y_test):
    sgd.fit(X_train, y_train)
    param_distribs = [{
        'loss': ['hinge', 'log', 'modified_huber'],
        'alpha': [0.0001, 0.00001, 0.000001]
        }]
    randomized_search = RandomizedSearchCV(
            sgd,
            param_distribs,
            cv=5,
            verbose=3,
            n_jobs=-1
            )
    randomized_search.fit(X_train, y_train)
    y_pred = randomized_search.predict(X_test)
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_
    accuracy = accuracy_score(y_test, y_pred)
    with open('sgd-randomized-search', 'w') as f:
        f.write(f'SGD Best Parameter: {best_params}\n')
        f.write(f'SGD Best Score: {best_score}')

    with open('sgd-accuracy-score', 'w') as f:
        f.write(f'SGD Accuracy Score: {accuracy}')


def forest_randomizedsearch(forest, X_train, y_train, X_test, y_test):
    forest.fit(X_train, y_train)
    param_distribs = [{
        'n_estimators': [3, 10, 30],
        'criterion': ['gini', 'entropy'],
        "max_features": [2, 4, 6, 8],
        }]
    randomized_search = RandomizedSearchCV(
            forest,
            param_distribs,
            cv=5,
            verbose=3,
            n_jobs=-1
            )
    randomized_search.fit(X_train, y_train)
    y_pred = randomized_search.predict(X_test)
    best_params = randomized_search.best_params_
    best_score = randomized_search.best_score_
    accuracy = accuracy_score(y_test, y_pred)
    with open('forest-randomized-search', 'w') as f:
        f.write(f'Random Forest Best Parameter: {best_params}\n')
        f.write(f'Random Forest Best Score: {best_score}')

    with open('forest-accuracy-score', 'w') as f:
        f.write(f'Random Forest Accuracy Score: {accuracy}')


if __name__ == '__main__':
    Fire(main)
