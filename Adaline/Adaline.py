import numpy as np
from numpy.linalg import inv


class AdalineRegressor:
    def __init__(self):
        self.w = None

    def fit(self, X_train, Y_train):
        self.w = np.matmul(inv(np.matmul(X_train.T, X_train)), np.matmul(X_train.T, Y_train))

    def predict(self, X_test):
        return np.matmul(X_test, self.w)

    def evaluation(self, X_test, Y_test):
        subtract = np.abs(Y_test - np.matmul(X_test, self.w))
        average = np.mean(subtract)

        mae = np.mean(subtract)
        mse = np.mean(subtract ** 2)

        return average, mae, mse
