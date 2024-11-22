#-------------#
# ARIMA Model #
#-------------#
import numpy as np
import pandas as pd
from data_cleaning import working_df


arima_df = working_df.copy()

#| Convert purchase_timestamp to datetime
arima_df['order_purchase_timestamp'] = pd.to_datetime(arima_df['order_purchase_timestamp'])

#| Ensure the data is sorted by timestamp
arima_df = arima_df.sort_values('order_purchase_timestamp')

#| Set the timestamp as the index
arima_df.set_index('order_purchase_timestamp', inplace=True)

#| Aggregate sales data by day (you can change 'D' to 'M' for monthly aggregation)
sales_data = arima_df.resample('W').agg({'payment_value': 'sum'})

#| Drop rows with missing values if necessary
sales_data = sales_data.dropna()

import matplotlib.pyplot as plt

#| Plot the time series data to inform filtering
plt.figure(figsize=(12, 6))
plt.plot(sales_data, label='Weekly Aggregated Revenue ($)')
plt.title('Sales Time Series')
plt.xlabel('Date')
plt.ylabel('Weekly Aggregated Revenue ($)')
plt.legend()
plt.show() # No data between late 2016 and early 2017, so trim it. Huge spike in late Nov 2017.

#| Trim data to remove 2016 entries.
#| Define the cutoff date
cutoff_date = '2017-01-01'

#| Filter the DataFrame to include only rows with indexes >= cutoff_date
sales_filtered = sales_data[sales_data.index >= cutoff_date]

#| Check data for stationarity
from statsmodels.tsa.stattools import adfuller # install as pip install statsmodels

result = adfuller(sales_filtered['payment_value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1]) # p = 0.33 so does not need differencing

#| Determine ARIMA orders
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plot_acf(sales_filtered, ax=plt.gca())
plt.subplot(122)
plot_pacf(sales_filtered, ax=plt.gca())
plt.show()

from pmdarima import auto_arima

#| Check values from auto arime function concur with plots
#| Fit auto_arima model
model = auto_arima(sales_filtered['payment_value'], seasonal=False, stepwise=True, trace=True)

#| Print the best model parameters
print(model.summary()) # SARIMAX(4, 1, 1)

#| Does log transforming data increase accuracy?
sales_filtered['log_payment_value'] = np.log(sales_filtered['payment_value'])
sales_filtered.dropna()
#| Remove 0s
sales_filtered = sales_filtered[sales_filtered['payment_value'] > 0]

#| Re run model on logged data
log_model = auto_arima(sales_filtered['log_payment_value'], seasonal=False, stepwise=True, trace =True)
print(log_model.summary()) # SARIMAX(0,2,1)

#| Test accuracy of models against real world data

#| Invoke train test split in the data
train_size = int(len(sales_filtered) * 0.8)  # Use 80% for training

#| Split data
train, test = sales_filtered[:train_size], sales_filtered[train_size:]
print(len(train)) # 69 (weeks worth of transcations)
print(len(test)) # 18 (weeks worth of transactions)

#| Retrain arima on training data of non-logged var
train_model_nlog = auto_arima(train['payment_value'], seasonal=True, m =12,  stepwise=True, trace=True)
print(train_model_nlog.summary()) 

#| Retrain arima on training data of logged var
train_model_log = auto_arima(train['log_payment_value'], seasonal=True, m = 12, stepwise=True, trace=True)
print(train_model_log.summary()) 

#| Forecast future values
forecast_length = len(test)
nlog_forecast, nlog_conf_int = train_model_nlog.predict(n_periods=forecast_length, return_conf_int=True)
log_forecast, log_conf_int = train_model_log.predict(n_periods=forecast_length, return_conf_int=True)

#| Create a DataFrame for forecasts
forecast_index = test.index
nlog_forecast_df = pd.DataFrame(nlog_forecast, index=forecast_index, columns=['forecast'])
log_forecast_df = pd.DataFrame(log_forecast, index=forecast_index, columns=['forecast'])
#| Transform log predictions back to non-logged values
log_forecast_df['forecast'] = np.exp(log_forecast_df['forecast']) 


#| Evaluate Models
from sklearn.metrics import mean_squared_error, mean_absolute_error
actual = test['payment_value']

#| Compute evaluation metrics (not logged)
nlog_mse = mean_squared_error(actual, nlog_forecast_df['forecast'])
nlog_mae = mean_absolute_error(actual, nlog_forecast_df['forecast'])
nlog_rmse = np.sqrt(nlog_mse)

print(f'MSE: {nlog_mse}')
print(f'MAE: {nlog_mae}')
print(f'RMSE: {nlog_rmse}') # 12475.49

#| Compute evaluation metrics (logged)
log_mse = mean_squared_error(actual, log_forecast_df['forecast'])
log_mae = mean_absolute_error(actual, log_forecast_df['forecast'])
log_rmse = np.sqrt(log_mse)

print(f'MSE: {log_mse}')
print(f'MAE: {log_mae}')
print(f'RMSE: {log_rmse}') # 33134.49

#| Plot actual vs forecasted values
#| Filter actual data to only show during the forecast period
actual_forecast_period = test.loc[nlog_forecast_df.index]

#| Plot actual data during forecast period (in blue)
plt.plot(train.index, train['payment_value'], label='Actual Data')
plt.plot(actual_forecast_period.index, actual_forecast_period['payment_value'], label='Actual Data (Forecast Period)', color='blue')
plt.plot(nlog_forecast_df.index, nlog_forecast_df['forecast'], color='red', label='Non-Log Transformed Forecast')
plt.fill_between(nlog_forecast_df.index, 
                 nlog_conf_int[:, 0], nlog_conf_int[:, 1], color='red', alpha=0.3)
plt.plot(log_forecast_df.index, log_forecast_df['forecast'], color='green', label='Log Transformed Forecast')
plt.fill_between(log_forecast_df.index, 
                 log_conf_int[:, 0], log_conf_int[:, 1], color='green', alpha=0.3)
plt.title('ARIMA Forecast vs Actual Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()