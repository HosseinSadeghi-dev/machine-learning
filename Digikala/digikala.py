import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from arabic_reshaper import reshape
from bidi.algorithm import get_display


def en_to_fa(num, formatter='%1.1f%%'):
    num_as_string = formatter % num
    mapping = dict(list(zip('0123456789.%', '۰۱۲۳۴۵۶۷۸۹.%')))
    return ''.join(mapping[digit] for digit in num_as_string)


digikala_data = pd.read_csv('csv/digikala-orders.csv')
digikala_data['InvoiceDate'] = pd.to_datetime(digikala_data['DateTime_CartFinalize'], errors='coerce')
digikala_data['InvoiceYearMonth'] = digikala_data['InvoiceDate'].map(lambda date: 100 * date.year + date.month)
revenue = digikala_data.groupby(['InvoiceYearMonth']).sum().reset_index()
plt.figure(figsize=(25, 6))
plt.plot(revenue['InvoiceYearMonth'].to_numpy(dtype='str'), revenue['Quantity_item'], marker='o')
plt.grid()

city = digikala_data.groupby(['city_name_fa']).sum().reset_index()
city['ratio'] = city['Quantity_item'] / city['Quantity_item'].sum() * 100
new_row = pd.DataFrame(
    data={'city_name_fa': ['دیگر'], 'ratio': city.query("ratio < 1.5")['ratio'].sum()})
city = pd.concat([city, new_row])
matplotlib.rc('font', **{'family': 'IranSans', 'size': 16})
labels = city.query("ratio > 1.5")['city_name_fa']
sizes = city.query("ratio > 1.5")['ratio']
persian_labels = [get_display(reshape(label)) for label in labels]

plt.figure(figsize=(15, 15))
plt.pie(sizes, labels=persian_labels, autopct=en_to_fa)
