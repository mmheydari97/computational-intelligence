import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


TEST_PERCENT = 0.3
EPOCHS = 4000
LR = 3


df = pd.read_csv("./data.csv", names=["X1", "X2", "Y"])

X = np.array(df[["X1", "X2"]].values)
y = np.array(df["Y"].to_list())


def plot_dataset(x, y_):
    plt.scatter(x[y_ == 0][:, 0], x[y_ == 0][:, 1])
    plt.scatter(x[y_ == 1][:, 0], x[y_ == 1][:, 1])
    plt.show()


# ------------------------- 1 -------------------------
plot_dataset(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_PERCENT, random_state=0, shuffle=True)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dsig(x):
    s = sigmoid(x)
    return (1-s)*s


def cost(y_, yh_):
    return np.sum(0.5*(y_-yh_)**2)


# ------------------------- Part One -------------------------
# initialize
W = np.random.normal(0, 1, (2, 1))
B = np.random.normal(0, 1, (1, 1))


# ------------------------- 3 -------------------------
for e in range(EPOCHS):
    # prediction
    yh = sigmoid(X_train.dot(W) + B)

    if e % 200 == 0:
        print("epoch: {}, training loss: {}".format(e+1, cost(y_train, yh)))

    # calculate differentials
    z = ((y_train - yh.flatten()) * dsig(X_train.dot(W) + B).flatten()).reshape(1, -1)

    dB = np.mean(z)
    dW = (z.dot(X_train)) / X_train.shape[0]

    # update weights
    W += LR*dW.T
    B += LR*dB

print("Training 1 Done.")
# ------------------------- 4 -------------------------
# test
y_pr = sigmoid(X_test.dot(W) + B)
y_pr = np.array(np.floor(y_pr*2), dtype=np.int).flatten()

plot_dataset(X_test, y_pr)

# initialize
V = np.random.normal(0, 1, (2, 1))
W = np.random.normal(0, 1, (2, 1))
U = np.random.normal(0, 1, (2, 1))
B0 = np.random.normal(0, 1, (1, 1))
B1 = np.random.normal(0, 1, (1, 1))
B2 = np.random.normal(0, 1, (1, 1))

for e in range(EPOCHS):

    z0 = sigmoid(X_train.dot(W) + B0)
    z1 = sigmoid(X_train.dot(V) + B1)
    Z = np.concatenate((z0, z1), axis=1)

    # prediction
    yh = sigmoid(Z.dot(U) + B2)

    if e % 200 == 0:
        print("epoch: {}, training loss: {}".format(e+1, cost(y_train, yh)))

    # calculate differentials
    zu = (y_train - yh.flatten()) * dsig(Z.dot(U) + B2).flatten()
    dU = (zu.reshape(1, -1).dot(Z)).reshape(2, 1)/X_train.shape[0]
    dB2 = np.mean(zu)

    zw = ((U[0]*zu) * dsig(X_train.dot(W) + B0).flatten())
    zv = ((U[1]*zu) * dsig(X_train.dot(V) + B1).flatten())

    dW = (zw.dot(X_train)).reshape(2, 1)/X_train.shape[0]
    dB0 = np.mean(zw)
    dV = (zv.dot(X_train)).reshape(2, 1)/X_train.shape[0]
    dB1 = np.mean(zv)

    # update weights
    W += LR*dW
    U += LR*dU
    V += LR*dV

    B0 += LR*dB0
    B1 += LR*dB1
    B2 += LR*dB2

print("Training 2 Done.")
z0 = sigmoid(X_test.dot(W) + B0)
z1 = sigmoid(X_test.dot(V) + B1)
Z = np.concatenate((z0, z1), axis=1)

# test
y_pr = sigmoid(Z.dot(U) + B2)
y_pr = np.array(np.floor(y_pr*2), dtype=np.int).flatten()
plot_dataset(X_test, y_pr)
