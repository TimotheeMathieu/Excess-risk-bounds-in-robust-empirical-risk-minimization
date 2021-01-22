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



def errorbar(x, y, err, legend, color):
    plt.plot(x, y, label=legend, color=color)
    plt.xlabel("$\ln(\Delta)$")
    plt.ylabel("$\ln(MSE)$")
    plt.plot(x, err[0], linestyle="--", color=color)
    plt.plot(x, err[1], linestyle="--", color=color)
    plt.fill_between(x, err[0], err[1], alpha=0.25, color=color)


def error(x, axis=0):
    return np.vstack([np.percentile(x, 25, axis=axis), np.percentile(x, 75, axis=axis)])

def plot_figure():

    clf = LogisticRegression(C=1e3)

    M = 50

    s = score(clf, 100, 1000, M, 0)

    Ks = np.arange(3, 80, 10)
    reshom = []
    resmom = []

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

    plt.subplot(2, 2, 3)
    reshom = []
    resmom = []
    s = score(clf, 1000, 1000, M, 0)

    for K in tqdm(Ks):
        clfmom = classifier(K=int(K), Delta=0, beta=0, eta0=1)
        clfhom = classifier(K=int(K), Delta=1, eta0=1, beta=0)
        reshom += [score(clfhom, 1000, 1000, M, 0)]
        resmom += [score(clfmom, 1000, 1000, M, 0)]


    Ks = np.arange(3, 800, 100)

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
