import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
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

#Stationary data test
# print("Results of Dickey-Fuller Test:")
# dftest = adfuller(group_by_month['sales'])
# dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
# for key, value in dftest[4].items():
#     dfoutput['Critical Value (%s)'%key] = value
# print(dfoutput)

df_diff = group_by_month.diff().diff(12).dropna()
print(df_diff.info())

# plot_acf(df_diff)
# plt.savefig("diff_acf.png")
# plot_pacf(df_diff)
# plt.savefig("diff_pacf.png")
#
# #Stationary diff data test
# print("Results of Dickey-Fuller Test:")
# dftest = adfuller(df_diff['sales'])
# dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
# for key, value in dftest[4].items():
#     dfoutput['Critical Value (%s)'%key] = value
# print(dfoutput)
