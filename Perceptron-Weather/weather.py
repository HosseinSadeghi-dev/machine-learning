import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Perceptron.Perceptron_W import PerceptronWeather

data = pd.read_csv('csv/weatherHistory.csv')
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], utc=True)
data['Hour'] = data['Formatted Date'].dt.hour
data['Day'] = data['Formatted Date'].dt.day
data['Month'] = data['Formatted Date'].dt.month
data['Year'] = data['Formatted Date'].dt.year
data['DayOfYear'] = data['Formatted Date'].dt.dayofyear

temperature_average = data.groupby(['DayOfYear', 'Year'])['Temperature (C)'].mean().reset_index()

fig = plt.figure(figsize=(27, 15))
plt.scatter(temperature_average['DayOfYear'], temperature_average['Temperature (C)'])
plt.xlabel('DayOfYear')
plt.ylabel('Temperature (C)')

X_train, X_test, Y_train, Y_test = train_test_split(temperature_average['DayOfYear'],
                                                    temperature_average['Temperature (C)'], test_size=0.1)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
X_train = X_train.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

model = PerceptronWeather()
loss = model.fit(X_train, Y_train)
# fig = plt.figure()
ax = fig.add_subplot()
ax.plot(loss)

model.evaluation(X_test, Y_test)
day = 45
t = model.predict(day)
t = t.reshape(1)
print(f'temperature on the {day}th day of the year: {t}')
