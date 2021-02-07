import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fire import Fire
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier #Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier #Stochastic Gradient Descent

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict

def main():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)
    sort_by_target(mnist)
    X, y = mnist['data'], mnist['target']
    some_digit = X[36000]
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
    sgd_clf.fit(X_train, y_train_5)

    # predict_with_threshold(sgd_clf, some_digit)

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    # plot_precision_recall_threshold(precisions, recalls, thresholds)
    # precision_higher_90(y_scores, y_train_5)

    fpr, tpr, threshold = roc_curve(y_train_5, y_scores)
    # plot_roc_curve(fpr, tpr)
    # save_fig('fpr-tpr-curve')

    # print(f'Area Under Curve: {roc_auc_score(y_train_5, y_scores)}')

    forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

    # random_forest_plot(forest_clf, X_train, y_train_5, fpr, tpr)
    # random_forest_precision_recall(forest_clf, X_train, y_train_5)

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]

def save_fig(fig_id, tight_layout=True):
    path = os.path.join('.', fig_id + ".png")
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary, interpolation='nearest')
    plt.axis('off')

def cross_validation_implementation(sgd_clf, X_train, y_train):
    skfolds = StratifiedKFold(n_splits=3)
    results = []

    for train_index, test_index in skfolds.split(X_train, y_train):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = (y_train[train_index])
        X_test_folds = X_train[test_index]
        y_test_folds = (y_train[test_index])

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        results.append(n_correct / len(y_pred))

    print(results)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.xlim([-700000, 700000])

def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b--", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def random_forest_plot(forest_clf, X_train, y_train_5, fpr, tpr):
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
    y_scores_forest = y_probas_forest[:, 1] #score = probability of positive class
    fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
    plt.plot(fpr, tpr, "b:", label="SGD")
    plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
    plt.legend(loc="lower right")
    save_fig('random-forest-curve')

def random_forest_precision_recall(forest_clf, X_train, y_train_5):
    y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3)
    print(f'Precision Random Forest: {precision_score(y_train_5, y_train_pred_forest)}')
    print(f'Recall Random Forest: {recall_score(y_train_5, y_train_pred_forest)}')

def plot_precision_recall_threshold(precisions, recalls, thresholds):
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plot_precision_vs_recall(precisions, recalls)
    save_fig('precision-recall')

def precision_higher_90(y_scores, y_train_5):
    y_train_pred_90 = (y_scores > 70000)
    print(f'Precision (almost or higher than 90%): {precision_score(y_train_5, y_train_pred_90)}')
    print(f'Recall: {recall_score(y_train_5, y_train_pred_90)}')

def predict_with_threshold(sgd_clf, some_digit):
    y_scores = sgd_clf.decision_function([some_digit])
    print(f'Decision function score: {y_scores}')
    threshold = 100000
    y_some_digit_pred = (y_scores > threshold)
    print(f'Prediction result (threshold: {threshold}): {y_some_digit_pred}\n')

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

if __name__ == '__main__':
    Fire(main)
