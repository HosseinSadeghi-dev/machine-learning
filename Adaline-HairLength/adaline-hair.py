import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

N = 50

x_male = np.random.normal(10, 2, N)
x_female = np.random.normal(50, 4, N)

label_female = np.zeros(N, dtype='int')
label_male = np.ones(N, dtype='int')

X = np.concatenate((x_female, x_male))
Y = np.concatenate((label_female, label_male))

plt.figure()
plt.scatter(X[0:N], Y[0:N], c='blue')
plt.scatter(X[N:], Y[N:], c='pink')
plt.xlabel('Hair')
plt.ylabel('Gender')

X = X.reshape(100, 1)
Y = Y.reshape(100, 1)

# false test No.1
plt.figure()
for i in range(5):
    m = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y)) + i
    y_pred = np.matmul(X, m)
    plt.scatter(X[0:N], Y[0:N], c='blue')
    plt.scatter(X[N:], Y[N:], c='pink')
    plt.plot(X, y_pred, c="green", lw=4)
plt.show()

# false test No.2
plt.figure()
for i in range(5):
    m = np.random.random(1)
    y_pred = np.matmul(X, m)
    plt.scatter(X[0:N], Y[0:N], c='blue')
    plt.scatter(X[N:], Y[N:], c='pink')
    plt.plot(X, y_pred, c="green", lw=4)
plt.show()

# right value
plt.figure()
m = np.matmul(
    inv(np.matmul(X.T, X)),
    np.matmul(X.T, Y)
)
y_pred = X * m
plt.plot(X, y_pred, c="green", lw=4)

plt.scatter(X[0:N], Y[0:N], c='blue')
plt.scatter(X[N:], Y[N:], c='pink')
plt.show()
