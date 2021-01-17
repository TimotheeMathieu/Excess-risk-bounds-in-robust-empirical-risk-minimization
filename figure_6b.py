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
