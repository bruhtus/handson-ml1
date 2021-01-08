import hashlib
import numpy as np
import pandas as pd

from fire import Fire

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

def main(data):
    pd.set_option('display.max_columns', None)
    housing = pd.read_csv(data)
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    #print(f'train set:\n{train_set}\n')
    #print(f'test set:\n{test_set}\n')

    #divide median_income by 1.5 to limit income categories
    housing['income_cat'] = np.ceil(housing['median_income']/1.5)
    #capped income catogories at 5.0
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    print(f'Count income categories:\n{housing["income_cat"].value_counts()}')
                                           

if __name__ == '__main__':
    Fire(main)
