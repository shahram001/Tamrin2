import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('i:/IOT/HW2/data.txt', sep=';', parse_dates=[['Date', 'Time']], na_values='?', low_memory=False)

data.rename(columns={'Date_Time': 'DateTime'}, inplace=True)
data.set_index('DateTime', inplace=True)

numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
data[numeric_cols] = data[numeric_cols].astype(float)

data.fillna(data.mean(), inplace=True)

X = data[['Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
y = data['Global_active_power']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

plt.figure(figsize=(14, 7))

y_test_sorted = y_test.sort_index()
y_pred_sorted = pd.Series(y_pred, index=y_test_sorted.index)

plt.plot(y_test_sorted.index, y_test_sorted.values, label="Actual", color='blue')
plt.plot(y_pred_sorted.index, y_pred_sorted.values, label="Predicted", color='red', linestyle='--')
plt.xlabel("DateTime")
plt.ylabel("Global Active Power (kW)")
plt.title("Actual vs Predicted Global Active Power Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
