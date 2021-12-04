# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:35:55 2021

@author: annap
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


#import white wines dataset
data = pd.read_csv('winequality-white.csv', sep=';')
data.describe()


#histograms
sn.histplot(data['quality'])
plt.show()
sn.histplot(data['fixed acidity'])
plt.show()
sn.histplot(data['volatile acidity'])
plt.show()
sn.histplot(data['citric acid'])
plt.show()
sn.histplot(data['residual sugar'])
plt.show()
sn.histplot(data['chlorides'])
plt.show()
sn.histplot(data['free sulfur dioxide'])
plt.show()
sn.histplot(data['total sulfur dioxide'])
plt.show()
sn.histplot(data['density'])
plt.show()
sn.histplot(data['pH'])
plt.show()
sn.histplot(data['sulphates'])
plt.show()
sn.histplot(data['alcohol'])
plt.show()

#boxplots: many outliers
sn.boxplot(data['quality'])
plt.show()
sn.boxplot(data['fixed acidity'])
plt.show()
sn.boxplot(data['volatile acidity'])
plt.show()
sn.boxplot(data['citric acid'])
plt.show()
sn.boxplot(data['residual sugar'])
plt.show()
sn.boxplot(data['chlorides'])
plt.show()
sn.boxplot(data['free sulfur dioxide'])
plt.show()
sn.boxplot(data['total sulfur dioxide'])
plt.show()
sn.boxplot(data['density'])
plt.show()
sn.boxplot(data['pH'])
plt.show()
sn.boxplot(data['sulphates'])
plt.show()
sn.boxplot(data['alcohol'])
plt.show()

col=data.columns.tolist()
col.remove('quality')
data_new=data[col]


#correlation matrix
corrMatrix = data_new.corr()
sn.heatmap(corrMatrix, annot = True)
plt.show()

#scatterplot
sn.pairplot(data_new)

#removing of outliers
data = data.loc[data['residual sugar'] < 40]
data = data.loc[data['free sulfur dioxide'] < 200]
data = data.loc[data['density'] < 1.005]
data = data.loc[data['fixed acidity'] < 12.5]
data = data.loc[data['citric acid'] < 1.5]

#new scatterplot to assess linear dependence
col=data.columns.tolist()
col.remove('quality')
data_new=data[col]
sn.pairplot(data_new)

#density is linearly correlated to alcohol and residual sugar --> we can avoid considering it.
col = data.columns.tolist()
col.remove('density')
data = data[col]

data['merged']=np.select([data['quality']<6,data['quality']==6, data['quality']>6],[0,1,2])
col.remove('quality')
data.to_csv(r'C:\Users\annap\Desktop\EPFL\SML\SML_project\whitewines_cleaned.csv')

