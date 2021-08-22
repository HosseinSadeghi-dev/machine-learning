import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import inv

N = 100
X = []
Y = []
for i in range(N):
    X.append(np.random.randint(60, 200, dtype=int))
    Y.append(random.uniform(0.25, 0.5) * X[i])
plt.scatter(X, Y)
X = np.array(X).reshape(100, 1)
Y = np.array(Y).reshape(100, 1)

# false tests (black color)
m1 = random.uniform(0.1, 1)
m2 = random.uniform(0.1, 1)
Y_pred1 = X * m1
Y_pred2 = X * m2
plt.scatter(X, Y)
plt.plot(X, Y_pred1, c='black')
plt.plot(X, Y_pred2, c='black')

# adaline output (red color)
m = np.matmul(inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
Y_pred = X * m
plt.scatter(X, Y)
plt.plot(X, Y_pred, c='red')

slope, intercept, r_value, p_value = stats.linregress(X, Y)

plt.plot(X, Y, 'o', label='original data')
plt.plot(X, intercept + slope * X, 'r', label='fitted line')
plt.legend()

plt.show()
