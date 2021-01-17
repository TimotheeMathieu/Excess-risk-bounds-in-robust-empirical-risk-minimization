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
