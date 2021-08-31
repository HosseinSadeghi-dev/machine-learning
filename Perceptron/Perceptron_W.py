import numpy as np


class PerceptronWeather:
    def __init__(self):
        self.W = np.random.rand(1, 1)
        self.b = np.random.rand(1, 1)
        self.lr = 0.000001
        self.all_w = []
        self.loss = []

    def fit(self, x_train, y_train):
        for i in range(len(x_train)):
            y_pred = np.matmul(x_train[i], self.W) + self.b
            e = y_train[i] - y_pred
            self.W = self.W + e * self.lr * x_train[i]
            self.b = self.b + e * self.lr

            Y_pred = np.matmul(x_train, self.W)
            error = np.mean(np.abs(y_train - Y_pred))
            self.loss.append(error)

            self.all_w.append(self.W)

        np.save('../Perceptron-Weather/outfile_W.npy', self.all_w + self.b)

        return self.loss

    def predict(self, x):
        x = np.array(x)
        x = x.reshape(1, 1)
        return np.matmul(x, self.W) + self.b

    def evaluation(self, X, Y):
        subtract = np.abs(Y - np.matmul(X, self.W) + self.b)
        return np.mean(subtract)
