import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv("Data/sales_train.csv")

data['date'] = data['date'].str.replace(".", "")
data['date'] = pd.to_datetime(data['date'], format='%d%m%Y')

data = data[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']]

new_data = data[['date', 'item_id', 'item_cnt_day']].copy()

group_by_month = new_data.groupby(new_data['date'].dt.strftime('%B-%Y'))['item_cnt_day'].sum().sort_values()

group_by_month = pd.DataFrame({'date': group_by_month.index, 'sales': group_by_month.values})

group_by_month['date'] = pd.to_datetime(group_by_month['date'])

group_by_month.set_index('date')

ax = plt.gca()

group_by_month.plot(kind='line', x='date', y='sales', ax=ax)
plt.savefig("Sales per month.png")

decomposition = seasonal_decompose(group_by_month['sales'], freq=12)
decomposition.plot()
plt.savefig("seasonal_decompose.png")

plot_acf(group_by_month)
plt.savefig("acf.png")
plot_pacf(group_by_month)
plt.savefig("pacf.png")