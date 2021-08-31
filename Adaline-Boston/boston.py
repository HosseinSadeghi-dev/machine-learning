import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets
from Adaline.Adaline import AdalineRegressor

data = datasets.load_boston()
N = len(data.data[:, 0])
X = np.array([data.data[:, 0], data.data[:, 5]]).T
Y = data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
model = AdalineRegressor()
model.fit(X_train, Y_train)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='green')
ax.set_xlabel('crime')
ax.set_ylabel('room')
ax.set_zlabel('Price')

Y_pred = model.predict(X_train)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

X_plan, Y_plan = np.meshgrid(np.arange(X_train[:, 0].min(), X_train[:, 0].max()),
                             np.arange(X_train[:, 1].min(), X_train[:, 1].max()))

Z = X_plan * model.w[0] + Y_plan * model.w[1]

myCMap = plt.get_cmap('gist_earth')
surf = ax.plot_surface(X_plan, Y_plan, Z, cmap=myCMap, alpha=0.5)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='green')
ax.set_xlabel('crime')
ax.set_ylabel('room')
ax.set_zlabel('Price')
plt.show()

result = model.evaluation(X_test, Y_test)
print('evaluation: ', result[0], ' mae: ', result[1], ' mse: ', result[2])
