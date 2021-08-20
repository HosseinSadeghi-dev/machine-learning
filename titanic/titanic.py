import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')

plt.hist(data.loc[(data['Sex'] == 'female') & (data['Age']), 'Age'], bins=30, density=True, color='tab:red',
         label='female')
plt.hist(data.loc[(data['Sex'] == 'male') & (data['Age']), 'Age'], bins=30, density=True, color='tab:blue',
         label='male')
plt.legend(frameon=False)
plt.xlabel('age')

plt.hist(data.loc[(data['Pclass'] == 1) & (data['Fare']), 'Fare'], bins=5, density=True, color='tab:red',
         label='Pclass 1')
plt.hist(data.loc[(data['Pclass'] == 2) & (data['Fare']), 'Fare'], bins=5, density=True, color='tab:green',
         label='Pclass 2')
plt.hist(data.loc[(data['Pclass'] == 3) & (data['Fare']), 'Fare'], bins=5, density=True, color='tab:red',
         label='Pclass 3')

plt.legend(frameon=False)
plt.xlabel('fare')
