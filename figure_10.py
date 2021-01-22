import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RepeatedKFold
import time
from sklearn.preprocessing import StandardScaler


names = pd.read_csv("crime_names.txt", sep=";").columns
df = pd.read_csv("CommViolPredUnnormalizedData.txt", names=names)

features = ["perCapInc", "population", "medFamInc", "HousVacant", "LandArea"]
target = "murders"

# Dump all missing data rows
df = df.dropna(axis="rows")

X = np.array(df[features])
y = np.array(df[target])
n = len(X)

std = StandardScaler()
X = std.fit_transform(X)


def score(clf, X, y, param_name, param, train_index, test_index):
    clf.set_params(**{param_name: param})
    clf.fit(X[train_index], y[train_index])
    pred = clf.predict(X[test_index])
    res = (y[test_index] - pred) ** 2
    # q=int(len(res)*0.1)
    return np.mean(np.sort(res))


def scores(
    clf, X, y, param_name, param_range, n_splits=100, n_repeat=1, n_jobs=10, verbose=10
):
    global total_n_jobs
    global init_time
    init_time = time.time()
    total_n_jobs = n_splits * n_repeat * len(param_range)
    rskf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeat)
    res = [
        score(clf, X, y, param_name, param, train_index, test_index)
        for train_index, test_index in rskf.split(X, y)
        for param in param_range
    ]
    params = [
        param for train_index, test_index in rskf.split(X, y) for param in param_range
    ]
    res = np.array(res).reshape([n_splits * n_repeat, len(param_range)]).T
    return res


n_splits = 500
# n_splits needs to be large for robustness.

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
s2 = scores(lr, X, y, "fit_intercept", [True], n_splits=n_splits)

from sklearn.linear_model import HuberRegressor

hr = HuberRegressor()
shub = scores(hr, X, y, "fit_intercept", [True], n_splits=n_splits, verbose=0)

from estimators import regressor

def plot_figure():

    reg = regressor(K=15, Delta=1e2, shuffle=True, epochs=30, beta=0)
    s1 = scores(reg, X, y, "K", [450], n_splits=n_splits, verbose=0, n_jobs=10)


    sns.distplot(np.log(s1.T), label="(A2)", norm_hist=True)
    sns.distplot(np.log(s2.T), label="OLS", norm_hist=True)
    sns.distplot(np.log(shub.T), label="Huber", norm_hist=True)

    plt.legend()
    plt.show()
