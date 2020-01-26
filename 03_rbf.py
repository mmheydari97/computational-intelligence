from fcm import FCM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split


class RBF:
    def __init__(self, gamma=0.1):
        self.gamma = gamma
        self.G = None
        self.C = None
        self.W = None
        self.fcm = None

    def fit(self, k, X_train, y_train):
        n, m = X_train.shape
        self.fcm = FCM(n_clusters=k)
        fcm = self.fcm.fit(X_train)
        V = fcm.centers
        U = fcm.u
        self.G = np.ndarray(shape=(n, k))
        self.C = np.zeros(shape=(k, m, m))
        for i in range(k):
            sm = 0
            for j in range(n):
                diff = np.array(X_train[j] - V[i]).reshape(-1, 1)
                self.C[i] += (U[j, i]**m)*diff*(diff.transpose())
                sm += U[j, i]**m
            self.C[i] /= sm
        self.C = np.array([np.linalg.inv(c) for c in self.C])

        for i in range(k):
            for j in range(n):
                diff = np.array(X_train[j] - V[i]).reshape(-1, 1)
                self.G[j, i] = np.exp(-self.gamma * (diff.transpose().dot(self.C[i])).dot(diff))

        ohe = OneHotEncoder(sparse=False)
        Y = ohe.fit_transform(y_train)
        self.W = np.linalg.inv(self.G.T.dot(self.G)).dot(self.G.T).dot(Y)
        return self

    def predict(self, X_test):
        n, m = X_test.shape
        V = self.fcm.centers
        U = self.fcm.predict(X_test)
        k = self.fcm.n_clusters
        G = np.ndarray(shape=(n, k))
        C = np.zeros(shape=(k, m, m))
        for i in range(k):
            sm = 0
            for j in range(n):
                diff = np.array(X_test[j] - V[i])
                C[i] += (U[j, i] ** m) * diff * (diff.transpose())
                sm += U[j, i]**m
            C[i] /= sm
        C = np.array([np.linalg.pinv(c) for c in C])
        for i in range(k):
            for j in range(n):
                diff = np.array(X_test[j] - V[i])
                G[j, i] = np.exp(-self.gamma * (diff.transpose().dot(C[i])).dot(diff))

        y_pred = np.argmax(G.dot(self.W), axis=1)
        return y_pred

    def get_accuracy(self, y_test, y_pred):
        return np.mean(np.equal(y_test.flatten(), y_pred.flatten()))


if __name__ == "__main__":
    df = pd.read_csv("2clstrain1200.csv", header=None, names=["x1", "x2", "y"])
    # df = pd.read_excel("5clstrain1500.xlsx", header=None, names=["x1", "x2", "y"])
    X = df.drop("y", axis=1).to_numpy()
    Y = df[["y"]].astype("str")
    oe = OrdinalEncoder(dtype=np.int)
    Y = oe.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    fig = plt.figure(figsize=(20, 5))
    i = 1
    for gm in [0.05, 0.1, 0.5, 1]:
        acc_train = []
        acc_test = []
        for nc in range(2, 10):
            # acc decreases after 10
            rbf = RBF(gm)
            rbf = rbf.fit(nc, X_train, y_train)
            y_pred = rbf.predict(X_train)
            acc_train.append(rbf.get_accuracy(y_train, y_pred))
            y_pred = rbf.predict(X_test)
            acc_test.append(rbf.get_accuracy(y_test, y_pred))

        ax = fig.add_subplot(1, 4, i)
        ax.plot(range(2, 10), acc_train)
        ax.plot(range(2, 10), acc_test)
        plt.legend(["train acc", "test acc"])
        i += 1

    plt.show()

    rbf = RBF(0.5)
    rbf = rbf.fit(4, X_train, y_train)
    y_pred = rbf.predict(X_test)
    print(rbf.get_accuracy(y_test, y_pred))

    plt.figure(figsize=(20, 12))
    # plot background
    xbg_min = min(X[:, 0])
    xbg_max = max(X[:, 0])
    ybg_min = min(X[:, 1])
    ybg_max = max(X[:, 1])

    xx = np.linspace(xbg_min, xbg_max, 200)
    yy = np.linspace(ybg_min, ybg_max, 120)
    mesh = np.transpose([np.tile(xx, len(yy)), np.repeat(yy, len(xx))])
    u_mesh = rbf.fcm.predict(mesh)
    y_mesh = np.argmax(u_mesh, axis=1)
    for i in range(u_mesh.shape[1]):
        idx = mesh[y_mesh == i]
        plt.scatter(idx[:, 0], idx[:, 1], marker="s", alpha=0.05, s=150)

    # plot data
    n_classes = np.unique(Y)
    for i in n_classes:
        idx = X[np.array(Y.flatten() == i)]
        plt.scatter(idx[:, 0], idx[:, 1])

    # plot missed
    fls = X_test[y_test.flatten() != y_pred]
    plt.scatter(fls[:, 0], fls[:, 1], marker="x", color="red", s=100)

    # plot centroids
    centroids = rbf.fcm.centers
    plt.scatter(centroids[:, 0], centroids[:, 1], marker="*", s=500)

    plt.show()





