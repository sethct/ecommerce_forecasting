#------------#
# LSTM Model #
#------------#
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#from data_cleaning import working_df

#| Set index and drop NAs introduced by weather data
lstm_df = working_df.copy()
lstm_df.set_index('date', inplace=True)
lstm_df = lstm_df.sort_index()
lstm_df = lstm_df.dropna()

#| Aggregate sales at the daily level
daily_sales = (
    lstm_df.groupby(['date', 'category_combined'])
    .agg({
        'payment_value': 'sum',  # Aggregate sales
        'is_holiday': 'first',   # Keep the first occurrence 
        'promotion_flag': 'first',
        'temp': 'first',
        'cloudcover': 'first',
        'precip': 'first'
    })
    .reset_index()
)

#| Set the index to date for further processing
daily_sales.set_index('date', inplace=True)

#| List to store performance metrics for each category
performance_data = []  # To store performance metrics for each category

#| Set up figure grid
fig, axs = plt.subplots(4, 3, figsize=(20, 15))  # 4x3 grid of plots
plt.subplots_adjust(hspace=0.4)  # Adjust the vertical spacing between plots

# Flatten the axs array for easier iteration
axs = axs.ravel()

#| Loop through each product category
for i, category in enumerate(daily_sales['category_combined'].unique()):
    if i >= 12:
        break  # Limit to 12 categories for the 4x3 grid

    print(f"Processing category: {category}")
    
    # Filter the dataframe for the current category
    category_df = daily_sales[daily_sales['category_combined'] == category].copy()
    
    # Scale sales and other numerical data
    scaler = MinMaxScaler(feature_range=(0, 1))
    category_df['sales_scaled'] = scaler.fit_transform(category_df[['payment_value']])
    category_df['is_holiday'] = category_df['is_holiday'].astype(int)

    # Define features and target
    features = category_df[['is_holiday', 'promotion_flag', 'temp', 'cloudcover', 'precip']].values
    target = category_df['sales_scaled'].values

    # Define time steps for LSTM (30 days)
    time_steps = 30
    X, y, dates = [], [], []
    for j in range(time_steps, len(features)):
        X.append(features[j-time_steps:j])
        y.append(target[j])
        dates.append(category_df.index[j])
    
    X, y = np.array(X), np.array(y)

    # Split into train and test sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    test_dates = dates[train_size:]

    # Create the LSTM model
    def create_model(units=50, dropout_rate=0.2, learning_rate=0.001):
        model = Sequential()
        model.add(LSTM(units=units, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    # Wrap model in KerasRegressor
    model = KerasRegressor(build_fn=create_model, verbose=0)

    # Define hyperparameter search space
    param_dist = {
        'model__units': [50, 100, 150],
        'model__dropout_rate': [0.2, 0.3, 0.4],
        'batch_size': [16, 32],
        'epochs': [10, 20],
    }

    # RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, cv=3, n_iter=10, n_jobs=-1)
    random_result = random_search.fit(X_train, y_train)

    # Get best model and predictions
    best_model = random_result.best_estimator_
    y_pred = best_model.predict(X_test)

    # Rescale the predictions and test values back to original sales values
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = mean_squared_error(y_test_rescaled, y_pred_rescaled, squared=False)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test_rescaled, y_pred_rescaled) * 100  # Multiply by 100 for percentage
    
    print(f"Root Mean Squared Error (RMSE) for {category}: {rmse}")
    print(f"Mean Absolute Percentage Error (MAPE) for {category}: {mape}")

    # Store performance data
    performance_data.append({
        'Category': category,
        'Best Params': random_result.best_params_,
        'RMSE': rmse,
        'MAPE': mape
    })

    # Plot predictions vs actual values in the appropriate subplot
    axs[i].plot(test_dates, y_test_rescaled.flatten(), label='Actual Sales', color='blue')
    axs[i].plot(test_dates, y_pred_rescaled.flatten(), label='Predicted Sales', color='red', linestyle='--')
    axs[i].set_title(f'{category}')
    axs[i].set_ylabel('Sales')

    # Only display x-axis labels for the bottom row
    if i < 9:
        axs[i].set_xticklabels([])

# Add x-axis labels only for the last row of plots
for ax in axs[8:]:
    ax.set_xlabel('Date')

# Adjust the layout for better spacing
plt.tight_layout()
plt.show()

#| Convert performance data to a DataFrame and print it
performance_df = pd.DataFrame(performance_data)
print(performance_df)