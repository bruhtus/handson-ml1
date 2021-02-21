import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fire import Fire
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier #Stochastic Gradient Descent
from sklearn.preprocessing import StandardScaler

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
    digit = y[36000]
    some_digit = X[36000]
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    # y_train_5 = (y_train == 5)
    # y_test_5 = (y_test == 5)

    # sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
    # sgd_clf.fit(X_train, y_train_5)

    # predict_with_threshold(sgd_clf, some_digit)

    # y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    # y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
    # precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

    # plot_precision_recall_threshold(precisions, recalls, thresholds)
    # precision_higher_90(y_scores, y_train_5)

    # fpr, tpr, threshold = roc_curve(y_train_5, y_scores)
    # plot_roc_curve(fpr, tpr)
    # save_fig('fpr-tpr-curve')

    # print(f'Area Under Curve: {roc_auc_score(y_train_5, y_scores)}')

    # forest_clf = RandomForestClassifier(n_estimators=10, random_state=42)

    # randomforest_plot_5detector(forest_clf, X_train, y_train_5, fpr, tpr)

    # sgd_clf.fit(X_train, y_train)
    # clf_predict_sgd(sgd_clf, digit, some_digit)
    # OneVsOne_classifier(X_train, y_train, some_digit)

    # forest_clf.fit(X_train, y_train)
    # clf_predict_forest(forest_clf, digit, some_digit)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    # print(f'Without scaler: {cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")}')
    # print(f'With scaler: {cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")}')

    # y_train_pred_scaled = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    # conf_mx = confusion_matrix(y_train, y_train_pred_scaled)
    # print(f'confusion matrix:\n{conf_mx}')
    # plt.matshow(conf_mx, cmap=plt.cm.gray)
    # save_fig('scaled-confusion-matrix', tight_layout=False)

    # row_sums = conf_mx.sum(axis=1, keepdims=True) #keepdims to keep dimensions of the array
    # norm_conf_mx = conf_mx / row_sums
    # np.fill_diagonal(norm_conf_mx, 0) #fill diagonal with 0s, keep only the errors
    # plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    # save_fig('compare-errors-rates', tight_layout=False)

    # check_individual_errors(X_train, y_train, y_train_pred_scaled, 3, 5)

    # y_train_large = (y_train >= 7)
    # y_train_odd = (y_train % 2 == 1)
    # y_multilabel = np.c_[y_train_large, y_train_odd] #two target labels

    # knn_clf = KNeighborsClassifier() #support multilabel classification
    # knn_clf.fit(X_train, y_multilabel)
    # print(f'The digit: {digit}\n')
    # print(f'>=7, odd: {knn_clf.predict([some_digit])}')
    # y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
    # print(f'KNN f1-score: {f1_score(y_train, y_train_knn_pred, average="macro")}')

    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test

    some_index = 5500
    plt.subplot(121); plot_digit(X_test_mod[some_index])
    plt.subplot(122); plot_digit(y_test_mod[some_index])
    save_fig('noisy-digit-example-plot')

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

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=mpl.cm.binary, **options)
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

def OneVsOne_classifier(X_train, y_train, some_digit):
    ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
    ovo_clf.fit(X_train, y_train)
    print(f'Predict: {ovo_clf.predict([some_digit])}')
    print(f'Length estimators: {len(ovo_clf.estimators_)}')

def randomforest_plot_5detector(forest_clf, X_train, y_train_5, fpr, tpr):
    random_forest_plot(forest_clf, X_train, y_train_5, fpr, tpr)
    random_forest_precision_recall(forest_clf, X_train, y_train_5)

def clf_predict_forest(clf, target_digit, image_digit):
    print(f'Real digit: {target_digit}')
    print(f'Predict: {clf.predict([image_digit])}')
    print(f'Classes: {clf.classes_}')
    print(f'Probabilities each class: {clf.predict_proba([image_digit])}')

def clf_predict_sgd(clf, target_digit, image_digit):
    print(f'Real digit: {target_digit}')
    print(f'Predict: {clf.predict([image_digit])}')
    print(f'Classes: {clf.classes_}')
    scores = clf.decision_function([image_digit])
    print(f'Index with highest score: {np.argmax(scores)}') #return the highest score

def check_individual_errors(X_train, y_train, y_train_pred, cl_a, cl_b):
    X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

    plt.figure(figsize=(8,8))
    plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
    plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
    plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
    plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
    save_fig('error-analysis-digits-plot', tight_layout=False)

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

if __name__ == '__main__':
    Fire(main)
