import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('COVID-19-time-series-clean-complete.csv')

data['Date'] = pd.to_datetime(data['Date'])
data['temp'] = data['Date'].map(lambda date: (100 * date.year) + date.month)
month = data.loc[(data['temp'] == 202004)]
_temp = month.sort_values(by=['Confirmed'], ascending=False)
x = month.groupby('Country/Region')['Confirmed'] \
    .sum() \
    .reset_index() \
    .sort_values(by=['Confirmed'], ascending=False)
x.head(8)

iran = data.loc[(data['Country/Region'] == 'Iran')]
plt.figure(figsize=(18, 9))
plt.plot(iran['Date'].to_numpy(dtype=str), iran['New Cases'], marker='O')
plt.plot(iran['Date'].to_numpy(dtype=str), iran['New RIPs'], marker='O')
