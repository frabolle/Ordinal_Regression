# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 11:10:36 2021

@author: annap
"""

"""
Created on Thu Dec  9 16:58:11 2021

@author: annap
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from statsmodels.miscmodels.ordinal_model import OrderedModel
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
sn.color_palette("colorblind")

# define dataset
#centers = [[-2, 0], [0, 0.5], [2, -1]]
#centers = [[0, 0], [0, 1], [0, 2]]
#centers = [[-2, 0], [-1, 0.5], [0, 0.5], [0.5,0.5]]
centers = [[-4, 0], [-2, 0], [0, 0], [2,0]]
#centers = [[0, 0], [0, 0.5], [0, 2], [0.5,2.5]]
X, y = make_blobs(n_samples=50, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)
X_new = X
y[17]=1
y[26]=0
y_new = y
data = pd.DataFrame(X,columns=['Cov1','Cov2'])
data['Class']=pd.DataFrame(y,columns=['Class'])
#17 deve diventare 1 e 26 deve diventare 0
# define the multinomial logistic regression model
h = 0.02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
plane = np.c_[xx.ravel(), yy.ravel()]

#non parallel ordered logistic model 
data = pd.DataFrame(X,columns=['Cov1','Cov2'])
data['Class']=pd.DataFrame(y,columns=['Class'])
data['first'] = (data['Class']>0).astype(int)
data['second']=(data['Class']>1).astype(int)
data['third']=(data['Class']>2).astype(int)

col=data.columns.to_list()
#col.remove(['Unnamed: 0','merged','first','second'])
col_list = [e for e in col if e not in ('Unnamed: 0','Class','first','second','third')]
X = data[col_list]
y = data['first']
log1=LogisticRegression(max_iter=450)
log1.fit(X,y)
predictions=log1.predict(X)

df_pred=X
df_pred['Y>0 true']=y
df_pred['Y>0 predicted']=predictions

X = data[col_list]
y = data['second']
log2=LogisticRegression(max_iter=450)
log2.fit(X,y)
predictions=log2.predict(X)
df_pred['Y>1 true']=y
df_pred['Y>1 predicted']=predictions

X = data[col_list]
y = data['third']
log3=LogisticRegression(max_iter=450)
log3.fit(X,y)
predictions=log3.predict(X)
df_pred['Y>2 true']=y
df_pred['Y>2 predicted']=predictions

error = df_pred[(df_pred['Y>0 predicted']==0) & (df_pred['Y>1 predicted']==1)]
error = df_pred[(df_pred['Y>1 predicted']==0) & (df_pred['Y>2 predicted']==1)]

df_pred['final prediction']=(df_pred['Y>0 predicted']==1).astype(int)+(df_pred['Y>1 predicted']==1).astype(int)+(df_pred['Y>2 predicted']==1).astype(int)

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_new, y_new)
yhat = model.predict(X)
model_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs')
model_ovr.fit(X_new, y_new)
yhat_ovr = model_ovr.predict(X_new)
mod_pom = OrderedModel(y_new,X_new,distr='logit')
model_pom = mod_pom.fit(method='bfgs')
num_of_thresholds = 2
mod_pom.transform_threshold_params(model_pom.params[-num_of_thresholds:])
yhat_pom_prob = model_pom.model.predict(model_pom.params, exog=X_new)
yhat_pom = yhat_pom_prob.argmax(1)



data_multinomial = pd.DataFrame(X_new,columns=['Cov1','Cov2'])
data_multinomial['Class']=pd.DataFrame(yhat,columns=['Class'])
data_ovr = pd.DataFrame(X_new,columns=['Cov1','Cov2'])
data_ovr['Class']=pd.DataFrame(yhat_ovr,columns=['Class'])
data_pom = pd.DataFrame(X_new,columns=['Cov1','Cov2'])
data_pom['Class']=pd.DataFrame(yhat_pom,columns=['Class'])


#palette colorblind
sn.scatterplot(x="Cov1",y="Cov2",data=data,hue="Class",palette = "colorblind")
plt.show()

fig, axes = plt.subplots(1,3,figsize=(15, 5))

sn.scatterplot(ax = axes[1], x="Cov1",y="Cov2",data=df_pred,hue="final prediction",palette = "colorblind")
axes[1].set_title('NP ordered logistic regression')
sn.scatterplot(ax = axes[0], x="Cov1",y="Cov2",data=data_multinomial,hue="Class",palette = "colorblind")
axes[0].set_title('Multinomial logistic regression')
sn.scatterplot(ax = axes[2], x="Cov1",y="Cov2",data=data_pom,hue="Class",palette = "colorblind")
axes[2].set_title('Ordered logistic regression')

# mat_multinomial = confusion_matrix(y,yhat)
# print(np.asarray(mat_multinomial))
# mat_ovr = confusion_matrix(y,yhat_ovr)
# print(np.asarray(mat_ovr))
# mat_pom = confusion_matrix(y,yhat_pom)
# print(np.asarray(mat_pom))


# fig, axes = plt.subplots(1,3,figsize=(15, 5))

# sn.heatmap(mat_multinomial,ax = axes[1], annot=True,cbar=False, cmap= "Greens")
# axes[1].set_title('Multinomial logistic regression');
# axes[1].set_xlabel('\nPredicted Class')
# axes[1].set_ylabel('Actual Class');
# sn.heatmap(mat_ovr,ax = axes[0],annot=True,cbar=False, cmap= "Greens")
# axes[0].set_title('OneVsRest logistic regression');
# axes[0].set_xlabel('\nPredicted Class')
# axes[0].set_ylabel('Actual Class');
# sn.heatmap(mat_pom,ax = axes[2],annot=True,cbar=False, cmap= "Greens")

# axes[2].set_title('Ordered logistic regression');
# axes[2].set_xlabel('\nPredicted Class')
# axes[2].set_ylabel('Actual Class');
## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False','True'])
# ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
