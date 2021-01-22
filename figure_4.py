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
def plot_figure():


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
