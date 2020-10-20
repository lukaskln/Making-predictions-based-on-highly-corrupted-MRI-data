#### Setup ####

import numpy as np
import pandas as pd
from scipy import stats
from itertools import repeat
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest

import xgboost as xgb

vColors = ["#049DD9", "#03A64A", "#F2AC29", "#F2CA80", "#F22929"]

# Import

X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv("y_train.csv")

X_test = pd.read_csv("X_test.csv")

# Distribution comparision between Train and Test Data

ks = []

for i in range(0, 832, 1):
    ks.append(stats.ks_2samp(X_train.iloc[:, i], X_test.iloc[:, i]).pvalue)

np.sum(pd.DataFrame(ks)[0] < 0.001)
pd.DataFrame(ks).index[pd.DataFrame(ks)[0] < 0.001]


x1 = X_train.iloc[:,200]
x2 = X_test.iloc[:,200]

hist_data = [x1, x2]

group_labels = ['Train', 'Test']

fig = ff.create_distplot(hist_data, group_labels, show_hist=True, colors=vColors, title = "Distribution between Train and Test Data")
fig.show()

#### Preprocessing testing ####

# NAs
X_train.isna().sum(axis=1)
X_train.fillna(X_train.median(), inplace=True)

X_train = X_train.iloc[:, 1:833]

# Scaling / Normalization
scaler = StandardScaler() # Can be changed to transformer

X_train = pd.DataFrame(
    scaler.fit_transform(
    X_train), 
    columns = X_train.columns)

# Response distribution

px.histogram(y_train, x="y", color_discrete_sequence=vColors, nbins = 100)

#### Feature Selection testing ####

# By LASSO / LARS Algorithm 

lsvc = Lasso(alpha = 0.1).fit(X_train, y_train["y"])
model = SelectFromModel(lsvc, prefit=True)
model.transform(X_train).shape

X_train = pd.DataFrame(model.transform(X_train))
X_test = pd.DataFrame(model.transform(X_test))

# By KV Test

X_train = X_train.loc[:, list(pd.DataFrame(ks)[0] > 0.001)]

# By (Pearson) Corr.
Corr = X_train.corrwith(y_train["y"], method = "pearson") # pearson vs spearman

fig = px.histogram(x=Corr, color_discrete_sequence=vColors[2:], nbins=100)

X_train = X_train.loc[:, (Corr > 0.1) | (Corr < -0.1)] 
X_train.shape

X = X_train["x47"].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y_train["y"])

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(X_train, x='x47',
                 y=y_train["y"], opacity=0.65, title="Correlation with Response")
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()

# By PCA

pca = PCA(n_components=20)
pca.fit(X_train)
dfComp = pd.DataFrame(pca.transform(X_train))

pca.explained_variance_

px.scatter(dfComp, x=0, y=1, opacity=0.65, color=y_train['y'],labels={
    "0": "First Principal Component",
    "1": "Second Principal Component",
    "color": "Age"
},
    title="Outliers as ring around Data")  # First PC has nice fade of the age

# By Importance Plot

model = xgb.XGBRegressor(colsample_bytree=0.4,
                         gamma=0,
                         learning_rate=0.07,
                         max_depth=3,
                         n_estimators=1000,
                         min_child_weight=1.5,
                         reg_alpha=0.75,
                         reg_lambda=0.45,
                         subsample=0.6,
                         seed=42)

model.fit(X_train, y_train["y"])

preds = model.predict(X_train)

r2_score(y_train["y"], preds)

plt.rcParams["figure.figsize"] = (10, 50)
xgb.plot_importance(model, importance_type="weight", show_values=False)
plt.show()

# Multicolinearity

CorrX = (X_train.corr() >= 0.5).sum().sum()/22968 

#### Outlier Detection testing ####

outliers_fraction = 0.001
nu_estimate = 0.95 * outliers_fraction + 0.04
auto_detection = svm.OneClassSVM(kernel="rbf", gamma = 0.00001, nu= nu_estimate )
auto_detection.fit(dfComp.iloc[:,[0,1]])
evaluation = auto_detection.predict(dfComp.iloc[:, [0, 1]])

X_train[evaluation ==-1].index.shape

px.scatter(dfComp, x=0, y=1, opacity=0.65, color=evaluation.astype(str), labels={
    "0": "First Principal Component",
    "1": "Second Principal Component",
    "color": "Outlier Selected"
},
    title="Outliers detected by SVC")


X_train = X_train[evaluation != -1]
y_train = y_train[evaluation != -1]

