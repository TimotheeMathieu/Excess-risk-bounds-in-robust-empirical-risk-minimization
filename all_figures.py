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

reg = regressor(K=15, Delta=1e2, shuffle=True, epochs=30, beta=0)
s1 = scores(reg, X, y, "K", [450], n_splits=n_splits, verbose=0, n_jobs=10)


sns.distplot(np.log(s1.T), label="(A2)", norm_hist=True)
sns.distplot(np.log(s2.T), label="OLS", norm_hist=True)
sns.distplot(np.log(shub.T), label="Huber", norm_hist=True)

plt.legend()
plt.show()
from estimators import classifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

## Make the dataset

np.random.seed(41)
centers = [-np.ones(2), np.ones(2)]

Xtrain, ytrain = datasets.make_blobs(
    n_samples=600, n_features=2, cluster_std=1.4, centers=centers, shuffle=True
)
for f in range(30):
    ind = int(np.floor(np.random.rand() * len(Xtrain)))
    Xtrain = np.vstack(
        [
            Xtrain[:ind],
            (0.1 * (np.random.randn(2)) + np.array([24, 8])).reshape(1, 2),
            Xtrain[ind:],
        ]
    )
    ytrain = np.hstack([ytrain[:ind], [0], ytrain[ind:]])

# Define and learn the classifiers

lr = LogisticRegression()
clf = classifier(K=91, Delta=0.01, eta0=30, epochs=300)

lr.fit(Xtrain, ytrain)
clf.fit(Xtrain, ytrain)


# Make the plot


plt.figure(figsize=(10, 3))
plt.subplot(1, 2, 1)  # LR plot


xx, yy = np.meshgrid(np.linspace(-5, 25, num=100), np.linspace(-5, 9, num=100))
Z = lr.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)


plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain)
plt.title("Logistic")
plt.subplot(1, 2, 2)  # Our plot

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)


plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain)
plt.title("Ours")
plt.show()
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from estimators import regressor
from matplotlib import pyplot as plt

# Make datasets


def make_data1(n_sample=600, nc=30):
    limit = int(n_sample * 2 / 3)
    X = np.random.uniform(-3, 3, size=[n_sample, 1])
    y = 10 * X.reshape(-1) + np.random.normal(size=n_sample)

    Xtrain = X[limit:]
    ytrain = y[limit:]

    Xtest = X[:limit]
    ytest = y[:limit]
    # Outliers

    for f in range(nc):
        ind = int(np.floor(np.random.rand() * len(Xtrain)))
        Xtrain = np.vstack(
            [
                Xtrain[:ind],
                (np.random.uniform(-3, 3, size=1)).reshape(1, 1),
                Xtrain[ind:],
            ]
        )
        ytrain = np.hstack(
            [ytrain[:ind], (0.1 * (np.random.randn(1)) + np.array([100])), ytrain[ind:]]
        )
    return Xtrain, ytrain, Xtest, ytest


def make_data2(n_sample=600, nc=30):
    limit = int(n_sample * 2 / 3)
    X = np.random.uniform(-3, 3, size=[n_sample, 1])
    y = 10 * X.reshape(-1) + np.random.normal(size=n_sample) + 20

    Xtrain = X[limit:]
    ytrain = y[limit:]

    Xtest = X[:limit]
    ytest = y[:limit]
    # Outliers

    for f in range(nc):
        ind = int(np.floor(np.random.rand() * len(Xtrain)))
        Xtrain = np.vstack(
            [
                Xtrain[:ind],
                (0.1 * (np.random.randn(1)) + np.array([24])).reshape(1, 1),
                Xtrain[ind:],
            ]
        )
        ytrain = np.hstack(
            [ytrain[:ind], (0.1 * (np.random.randn(1)) + np.array([24])), ytrain[ind:]]
        )
    return Xtrain, ytrain, Xtest, ytest


# First figure
plt.subplot(1, 2, 1)

## Draw dataset and fit
reg = LinearRegression()
hreg = HuberRegressor()

Xtrain, ytrain, Xtest, ytest = make_data1()
reg.fit(Xtrain, ytrain)
line_X = np.arange(Xtrain.min(), Xtrain.max())[:, np.newaxis]
line_y = reg.predict(line_X)

hreg.fit(Xtrain, ytrain)
hline_X = np.arange(Xtrain.min(), Xtrain.max())[:, np.newaxis]
hline_y = hreg.predict(line_X)

hom_c1 = regressor(K=91, eta0=0.01, Delta=1, beta=0)
hom_c1.fit(Xtrain, ytrain)
line_yh_c1 = hom_c1.predict(line_X)

# Make the plot

plt.scatter(Xtrain, ytrain)
plt.plot(line_X, line_y, color="red", linewidth=1, label="Linear regressor")
plt.plot(
    line_X,
    line_yh_c1,
    color="blue",
    linewidth=1,
    label="Our method [Alg 3], $\Delta=1$",
)
plt.plot(hline_X, hline_y, color="green", linewidth=1, label="Huber regressor")

plt.legend()


# Second figure
plt.subplot(1, 2, 2)

## Draw dataset and fit
reg = LinearRegression()
hreg = HuberRegressor()

Xtrain, ytrain, Xtest, ytest = make_data2()
reg.fit(Xtrain, ytrain)
line_X = np.arange(Xtrain.min(), Xtrain.max())[:, np.newaxis]
line_y = reg.predict(line_X)

hreg.fit(Xtrain, ytrain)
hline_X = np.arange(Xtrain.min(), Xtrain.max())[:, np.newaxis]
hline_y = hreg.predict(line_X)

hom_c1 = regressor(K=91, eta0=0.01, Delta=1, beta=0)
hom_c1.fit(Xtrain, ytrain)
line_yh_c1 = hom_c1.predict(line_X)

# Make the plot

plt.scatter(Xtrain, ytrain)
plt.plot(line_X, line_y, color="red", linewidth=1, label="Linear regressor")
plt.plot(
    line_X,
    line_yh_c1,
    color="blue",
    linewidth=1,
    label="Our method [Alg 3], $\Delta=1$",
)
plt.plot(hline_X, hline_y, color="green", linewidth=1, label="Huber regressor")

plt.legend()
plt.show()
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import clone
import numpy as np
from matplotlib import pyplot as plt
from estimators import regressor


def make_data(n_sample=500, nc=30):
    limit = int(n_sample * 2 / 3)
    X = np.random.uniform(-3, 3, size=[n_sample, 1])
    y = 10 * X.reshape(-1) + np.random.normal(size=n_sample) + 20

    Xtrain = X[limit:]
    ytrain = y[limit:]

    Xtest = X[:limit]
    ytest = y[:limit]
    # Outliers

    for f in range(nc):
        ind = int(np.floor(np.random.rand() * len(Xtrain)))
        Xtrain = np.vstack(
            [
                Xtrain[:ind],
                (0.1 * (np.random.randn(1)) + np.array([24])).reshape(1, 1),
                Xtrain[ind:],
            ]
        )
        ytrain = np.hstack(
            [ytrain[:ind], (0.1 * (np.random.randn(1)) + np.array([24])), ytrain[ind:]]
        )
    return Xtrain, ytrain, Xtest, ytest


def score(clf, M=300, N=500, nc=30):
    res = []
    for f in range(M):
        clf2 = clone(clf)
        Xtrain, ytrain, Xtest, ytest = make_data(n_sample=N, nc=nc)
        clf2.fit(Xtrain, ytrain)
        res += [mean_squared_error(clf2.predict(Xtest), ytest)]
    return res


def score_param(clf, param_name, param, M, N, nc):
    clf2 = clone(clf)
    clf2.set_params(**{param_name: param})
    return score(clf2, M, N, nc)


def scores(clf, param_name, param_range, M=50, N=900, nc=30):
    res = [score_param(clf, param_name, param, M, N, nc) for param in tqdm(param_range)]
    return res


def errorbar(x, y, err, legend, color):
    plt.plot(np.log(x), np.log(y), label=legend, color=color)
    plt.xlabel("$\ln(\Delta)$")
    plt.ylabel("$\ln(MSE)$")
    plt.plot(np.log(x), err[0], linestyle="--", color=color)
    plt.plot(np.log(x), err[1], linestyle="--", color=color)
    plt.fill_between(np.log(x), err[0], err[1], alpha=0.25, color=color)


def error(x, axis=0):
    return np.log(
        np.vstack([np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)])
    )


from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import clone
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
from estimators import regressor


def make_data(n_sample=500, nc=30):
    limit = int(n_sample * 2 / 3)
    X = np.random.uniform(-3, 3, size=[n_sample, 1])
    y = 10 * X.reshape(-1) + np.random.normal(size=n_sample) + 20

    Xtrain = X[limit:]
    ytrain = y[limit:]

    Xtest = X[:limit]
    ytest = y[:limit]
    # Outliers

    for f in range(nc):
        ind = int(np.floor(np.random.rand() * len(Xtrain)))
        Xtrain = np.vstack(
            [
                Xtrain[:ind],
                (0.1 * (np.random.randn(1)) + np.array([24])).reshape(1, 1),
                Xtrain[ind:],
            ]
        )
        ytrain = np.hstack(
            [ytrain[:ind], (0.1 * (np.random.randn(1)) + np.array([24])), ytrain[ind:]]
        )
    return Xtrain, ytrain, Xtest, ytest


def score(clf, M=300, N=500, nc=30):
    res = []
    for f in range(M):
        clf2 = clone(clf)
        Xtrain, ytrain, Xtest, ytest = make_data(n_sample=N, nc=nc)
        clf2.fit(Xtrain, ytrain)
        res += [mean_squared_error(clf2.predict(Xtest), ytest)]
    return res


def errorbar(x, y, err, legend, color):
    plt.plot(x, np.log(y), label=legend, color=color)
    plt.xlabel("$k$")
    plt.ylabel("$\ln(MSE)$")
    plt.plot(x, err[0], linestyle="--", color=color)
    plt.plot(x, err[1], linestyle="--", color=color)
    plt.fill_between(x, err[0], err[1], alpha=0.25, color=color)


def error(x, axis=0):
    return np.log(
        np.vstack([np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)])
    )


# M and epochs are smaller that what was used in the article because what was used in the article is
# rather long to compute.

Ks = np.arange(21, 180, 10)
res = []
for K in tqdm(Ks):
    clf = regressor(K=K, eta0=0.01, Delta=1, beta=0, epochs=100)
    res += [score(clf, M=50, N=900, nc=30)]
    print(K, np.log(np.median(np.array(res), axis=1)))
errorbar(Ks, np.median(res, axis=1), error(res, axis=1), "error", "b")

plt.show()
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import clone
import numpy as np
from matplotlib import pyplot as plt
from estimators import regressor


def make_data(n_sample=500, nc=30):
    limit = int(n_sample * 2 / 3)
    X = np.random.uniform(-3, 3, size=[n_sample, 1])
    y = 10 * X.reshape(-1) + np.random.normal(size=n_sample) + 20

    Xtrain = X[limit:]
    ytrain = y[limit:]

    Xtest = X[:limit]
    ytest = y[:limit]
    # Outliers

    for f in range(nc):
        ind = int(np.floor(np.random.rand() * len(Xtrain)))
        Xtrain = np.vstack(
            [
                Xtrain[:ind],
                (0.1 * (np.random.randn(1)) + np.array([24])).reshape(1, 1),
                Xtrain[ind:],
            ]
        )
        ytrain = np.hstack(
            [ytrain[:ind], (0.1 * (np.random.randn(1)) + np.array([24])), ytrain[ind:]]
        )
    return Xtrain, ytrain, Xtest, ytest


def score(clf, M=300, N=500, nc=30):
    res = []
    for f in range(M):
        clf2 = clone(clf)
        Xtrain, ytrain, Xtest, ytest = make_data(n_sample=N, nc=nc)
        clf2.fit(Xtrain, ytrain)
        res += [mean_squared_error(clf2.predict(Xtest), ytest)]
    return res


def score_param(clf, param_name, param, M, N, nc):
    clf2 = clone(clf)
    clf2.set_params(**{param_name: param})
    return score(clf2, M, N, nc)


def scores(clf, param_name, param_range, M=50, N=900, nc=30):
    res = [score_param(clf, param_name, param, M, N, nc) for param in tqdm(param_range)]
    return res


def errorbar(x, y, err, legend, color):
    plt.plot(np.log(x), np.log(y), label=legend, color=color)
    plt.xlabel("$\ln(\Delta)$")
    plt.ylabel("$\ln(MSE)$")
    plt.plot(np.log(x), err[0], linestyle="--", color=color)
    plt.plot(np.log(x), err[1], linestyle="--", color=color)
    plt.fill_between(np.log(x), err[0], err[1], alpha=0.25, color=color)


def error(x, axis=0):
    return np.log(
        np.vstack([np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)])
    )


from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn import clone
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, HuberRegressor
import numpy as np
from sklearn import linear_model, datasets
from matplotlib import pyplot as plt
from estimators import regressor


def make_data(n_sample=500, nc=30):
    limit = int(n_sample * 2 / 3)
    X = np.random.uniform(-3, 3, size=[n_sample, 1])
    y = 10 * X.reshape(-1) + np.random.normal(size=n_sample) + 20

    Xtrain = X[limit:]
    ytrain = y[limit:]

    Xtest = X[:limit]
    ytest = y[:limit]
    # Outliers

    for f in range(nc):
        ind = int(np.floor(np.random.rand() * len(Xtrain)))
        Xtrain = np.vstack(
            [
                Xtrain[:ind],
                (0.1 * (np.random.randn(1)) + np.array([24])).reshape(1, 1),
                Xtrain[ind:],
            ]
        )
        ytrain = np.hstack(
            [ytrain[:ind], (0.1 * (np.random.randn(1)) + np.array([24])), ytrain[ind:]]
        )
    return Xtrain, ytrain, Xtest, ytest


def score(clf, M=300, N=500, nc=30):
    res = []
    for f in range(M):
        clf2 = clone(clf)
        Xtrain, ytrain, Xtest, ytest = make_data(n_sample=N, nc=nc)
        clf2.fit(Xtrain, ytrain)
        res += [mean_squared_error(clf2.predict(Xtest), ytest)]
    return res


def score_param(clf, param_name, param, M, N, nc):
    clf2 = clone(clf)
    clf2.set_params(**{param_name: param})
    return score(clf2, M, N, nc)


def scores(clf, param_name, param_range, M=50, N=900, nc=30):
    res = [score_param(clf, param_name, param, M, N, nc) for param in tqdm(param_range)]
    return res


def errorbar(x, y, err, legend, color):
    plt.plot(np.log(x), np.log(y), label=legend, color=color)
    plt.xlabel("$\ln(\Delta)$")
    plt.ylabel("$\ln(MSE)$")
    plt.plot(np.log(x), err[0], linestyle="--", color=color)
    plt.plot(np.log(x), err[1], linestyle="--", color=color)
    plt.fill_between(np.log(x), err[0], err[1], alpha=0.25, color=color)


def error(x, axis=0):
    return np.log(
        np.vstack([np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)])
    )


# M is smaller that what was used in the article because what was used in the article is
# rather long to compute.
clf61 = regressor(K=61, eta0=1, Delta=1, beta=0, epochs=200, shuffle=False)
clf91 = regressor(K=91, eta0=1, Delta=1, beta=0, epochs=200, shuffle=False)
clf151 = regressor(K=151, eta0=1, Delta=1, beta=0, epochs=200, shuffle=False)
res61 = scores(clf61, "Delta", np.logspace(-3, 4.5, num=20), M=50, N=900)
res91 = scores(clf91, "Delta", np.logspace(-3, 4.5, num=20), M=50, N=900)
res151 = scores(clf151, "Delta", np.logspace(-3, 4.5, num=20), M=50, N=900)

Ds = np.logspace(-3, 4.5, num=20)
errorbar(Ds, np.median(res61, axis=1), error(res61, axis=1), "K=61", color="b")
errorbar(Ds, np.median(res91, axis=1), error(res91, axis=1), "K=91", color="r")
errorbar(Ds, np.median(res151, axis=1), error(res151, axis=1), "K=151", color="green")
plt.legend()
plt.show()
import numpy as np
from sklearn import  datasets
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from tqdm import tqdm
from estimators import classifier
from sklearn.preprocessing import StandardScaler
from sklearn import clone
from joblib import Parallel, delayed


def make_dataset(N=100, Nt=1000, Nc=3):
    X, y = datasets.make_moons(n_samples=N, noise=0.2)
    scal = StandardScaler()
    X = scal.fit_transform(X)

    perm = np.random.permutation(N)
    if Nc != 0:
        X[perm[-Nc:]] = np.vstack([np.zeros(Nc), 3 * np.ones(Nc)]).T
        y[perm[-Nc:]] = 1

    Xtest, ytest = datasets.make_moons(n_samples=Nt, noise=0.1)
    Xtest = scal.transform(Xtest)
    return X, y, Xtest, ytest


def score2(clf, N, Nt, Nc, r):
    np.random.seed(r)
    X, y, Xt, yt = make_dataset(N, Nt, Nc)
    clf2 = clone(clf)
    clf2.fit(X, y)
    return np.mean(clf2.predict(Xt) == yt)


def score(clf, N=100, Nt=1000, M=300, Nc=3):
    res = Parallel(n_jobs=5, prefer="threads")(
        delayed(score2)(clf, N, Nt, Nc, f + int(100 * np.random.uniform()))
        for f in np.arange(M)
    )
    return res


Ks = np.arange(3, 80, 6)


def errorbar(x, y, err, legend, color):
    plt.plot(x, y, label=legend, color=color)
    plt.xlabel("$\ln(\Delta)$")
    plt.ylabel("$\ln(MSE)$")
    plt.plot(x, err[0], linestyle="--", color=color)
    plt.plot(x, err[1], linestyle="--", color=color)
    plt.fill_between(x, err[0], err[1], alpha=0.25, color=color)


def error(x, axis=0):
    return np.vstack([np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)])


clf = LogisticRegression(C=1e3)

M = 300

s = score(clf, 100, 1000, M, 0)

Ks = np.arange(3, 80, 10)
reshom = []
resmom = []

# In this script we use momentum contrary to what is done for the article because without it the convergence is much harder to have.
# The reason we chose to put this script is 1) it would allow an interested user to tinker with it, without momentum changing the parameters would make the algo non convergent 2) this also allows for a much faster computation because we don't need a lot of epoch and a high M.

plt.subplot(2, 2, 1)

for K in tqdm(Ks):
    clfmom = classifier(K=int(K), Delta=0, beta=0, eta0=1)
    clfhom = classifier(K=int(K), Delta=1, eta0=1, beta=0)
    reshom += [score(clfhom, 100, 1000, M, 0)]
    resmom += [score(clfmom, 100, 1000, M, 0)]

errorbar(
    Ks,
    np.median(reshom, axis=1),
    error(reshom, axis=1),
    "Our estimator [Alg 3], $\Delta=1",
    color="b",
)
errorbar(
    Ks,
    np.median(resmom, axis=1),
    error(resmom, axis=1),
    "MOM estimator [Alg 4]",
    color="r",
)
errorbar(
    [Ks[0], Ks[-1]],
    [np.median(s), np.median(s)],
    np.hstack([error(s), error(s)]),
    "Logistic Regression",
    color="g",
)

plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("N=100, no outliers")
plt.legend()

plt.subplot(2, 2, 2)
reshom = []
resmom = []
s = score(clf, 100, 1000, M, 10)

for K in tqdm(Ks):
    clfmom = classifier(K=int(K), Delta=0, beta=0, eta0=5)
    clfhom = classifier(K=int(K), Delta=0.01, eta0=5, beta=0)
    reshom += [score(clfhom, 100, 1000, M, 10)]
    resmom += [score(clfmom, 100, 1000, M, 10)]

errorbar(
    Ks,
    np.median(reshom, axis=1),
    error(reshom, axis=1),
    "Our estimator [Alg 3], $\Delta=0.01",
    color="b",
)
errorbar(
    Ks,
    np.median(resmom, axis=1),
    error(resmom, axis=1),
    "MOM estimator [Alg 4]",
    color="r",
)
errorbar(
    [Ks[0], Ks[-1]],
    [np.median(s), np.median(s)],
    np.hstack([error(s), error(s)]),
    "Logistic Regression",
    color="g",
)

plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("N=100, 10 outliers")

plt.legend()
Ks = np.arange(3, 800, 100)

plt.subplot(2, 2, 3)
reshom = []
resmom = []
s = score(clf, 1000, 1000, M, 0)

for K in tqdm(Ks):
    clfmom = classifier(K=int(K), Delta=0, beta=0, eta0=0.1)
    clfhom = classifier(K=int(K), Delta=1, eta0=0.1, beta=0)
    reshom += [score(clfhom, 1000, 1000, M, 0)]
    resmom += [score(clfmom, 1000, 1000, M, 0)]

errorbar(
    Ks,
    np.median(reshom, axis=1),
    error(reshom, axis=1),
    "Our estimator [Alg 3], $\Delta=1",
    color="b",
)
errorbar(
    Ks,
    np.median(resmom, axis=1),
    error(resmom, axis=1),
    "MOM estimator [Alg 4]",
    color="r",
)
errorbar(
    [Ks[0], Ks[-1]],
    [np.median(s), np.median(s)],
    np.hstack([error(s), error(s)]),
    "Logistic Regression",
    color="g",
)

plt.xlabel("k")
plt.ylabel("accuracy")
plt.title("N=1000, no outliers")
plt.legend()

plt.subplot(2, 2, 4)
s = score(clf, 1000, 1000, M, 100)

reshom = []
resmom = []
for K in tqdm(Ks):
    clfmom = classifier(K=int(K), Delta=0, beta=0, eta0=0.1)
    clfhom = classifier(K=int(K), Delta=0.01, eta0=0.1, beta=0)
    reshom += [score(clfhom, 1000, 1000, M, 100)]
    resmom += [score(clfmom, 1000, 1000, M, 100)]

errorbar(
    Ks,
    np.median(reshom, axis=1),
    error(reshom, axis=1),
    "Our estimator [Alg 3], $\Delta=0.01",
    color="b",
)
errorbar(
    Ks,
    np.median(resmom, axis=1),
    error(resmom, axis=1),
    "MOM estimator [Alg 4]",
    color="r",
)
errorbar(
    [Ks[0], Ks[-1]],
    [np.median(s), np.median(s)],
    np.hstack([error(s), error(s)]),
    "Logistic Regression",
    color="g",
)

plt.xlabel("k")
plt.ylabel("accuracy")
plt.legend()
plt.title("N=1000, 100 outliers")
plt.show()
