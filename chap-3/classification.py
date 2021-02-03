import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fire import Fire
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier #Stochastic Gradient Descent

def main():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)
    sort_by_target(mnist)
    X, y = mnist['data'], mnist['target']
    some_digit = X[36000]
    X_test, y_test = X[60000:], y[60000:]
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X[shuffle_index], y[shuffle_index]

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    print(sgd_clf.predict([some_digit]))

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, 1) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[:60000] = mnist.data[reorder_test + 60000]
    mnist.target[:60000] = mnist.target[reorder_test + 60000]

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

if __name__ == '__main__':
    Fire(main)
