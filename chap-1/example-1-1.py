import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('module://drawilleplot')
import matplotlib.pyplot as plt
#from matplotlib_terminal import plt
from fire import Fire
from sklearn import linear_model, neighbors

def main():
    #load the data
    datapath = os.path.join('..', 'datasets', 'lifeat', '')
    oecd_bli = pd.read_csv(datapath + 'oecd_bli_2015.csv', thousands=',')
    gdp_per_capita = pd.read_csv(datapath + 'gdp_per_capita.csv',
                                thousands=',', delimiter='\t', encoding='latin1',
                                na_values='n/a')

    #prepare the data
    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
    X = np.c_[country_stats["GDP per capita"]]
    y = np.c_[country_stats["Life satisfaction"]]

    #visualize the data
    country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
    plt.show()
    #plt.show('block')

    #select linear model
    lin_reg_model = linear_model.LinearRegression()
    knn_model = neighbors.KNeighborsRegressor(n_neighbors=3)

    #train the model
    lin_reg_model.fit(X,y)
    knn_model.fit(X,y)

    #make prediction for cyprus
    #X_new = [[22587]] #cyprus
    X_new = [[11572]] #croatia
    print(f'Linear Regression: {lin_reg_model.predict(X_new)}')
    print(f'K-Nearest Neighbors: {knn_model.predict(X_new)}')

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

if __name__ == '__main__':
    Fire(main)
