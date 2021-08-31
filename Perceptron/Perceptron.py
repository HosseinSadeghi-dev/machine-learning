import numpy as np


class Perceptron:
    def __init__(self, w):
        self.w = w

    def predict(self, X_test):
        return np.matmul(X_test, self.w)

    def evaluation(self, X_test, Y_test):
        Y_predic = np.matmul(X_test, self.w)
        Y_predic[Y_predic > 0] = 1
        Y_predic[Y_predic < 0] = -1

        for i in range(len(X_test)):
            print('X_test[i] : ', X_test[i])
            print('w: ', self.w)

        Y_predic = np.matmul(X_test, self.w)
        subtract = np.abs(Y_test - Y_predic)
        average = np.mean(subtract)

        mae = np.mean(subtract)
        mse = np.mean(subtract ** 2)

        return average, mae, mse
