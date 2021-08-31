from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

tx_data = pd.read_csv('csv/digikala-orders.csv', encoding="latin1")
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['DateTime_CartFinalize'])
tx_data['InvoiceYearMonth'] = tx_data['InvoiceDate'].map(lambda date: 100 * date.year + date.month)
tx_revenue = tx_data.groupby(['InvoiceYearMonth']).sum().reset_index()
fig = plt.figure(figsize=(50, 12))
plt.plot(tx_revenue['InvoiceYearMonth'].to_numpy(dtype='str'), tx_revenue['Quantity_item'], marker='@')
plt.grid()
