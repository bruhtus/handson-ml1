import os
import numpy as np
import matplotlib.pyplot as plt

from fire import Fire
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift


def main():
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    mnist.target = mnist.target.astype(np.int8)

    sort_by_target(mnist)

    X, y = mnist['data'], mnist['target']

    digit = y[36000]
    some_digit = X[36000]

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    image = some_digit
    shifted_image_down = shift_image(image, 0, 5)
    shifted_image_left = shift_image(image, -5, 0)

    shifted_image_direction(image, shifted_image_down, shifted_image_left)


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


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode='constant')
    return shifted_image.reshape([-1])


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(".", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


def shifted_image_direction(image, shifted_image_down, shifted_image_left):
    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.title('original', fontsize=14)
    plt.imshow(image.reshape(28, 28),
               interpolation='nearest',
               cmap='Greys')

    plt.subplot(132)
    plt.title('shifted down', fontsize=14)
    plt.imshow(shifted_image_down.reshape(28, 28),
               interpolation='nearest',
               cmap='Greys')

    plt.subplot(133)
    plt.title('shifted left', fontsize=14)
    plt.imshow(shifted_image_left.reshape(28, 28),
               interpolation='nearest',
               cmap='Greys')

    save_fig('shifted-image')


if __name__ == '__main__':
    Fire(main)
