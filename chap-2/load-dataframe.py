import os
import pandas as pd
import matplotlib
matplotlib.use('module://drawilleplot')
import matplotlib.pyplot as plt

def load_housing_data(path='../datasets/housing'):
    csv_path = os.path.join(path, 'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(f'{housing.head()}\n')
print(f'{housing.info()}\n')
print(f'{housing.describe()}\n')
print(f"{housing['ocean_proximity'].value_counts()}\n")

housing.hist(bins=50, figsize=(20,15))
plt.show()
