'''
    THIS NOTEBOOK IS THE OLD VERSION OF THE CHURN ANALYSIS CORE
    A NEW VERSION HAS BEEN UPLOADED UNDER churn_analysis.py
    THIS OLD VERSION USES SIGMOID FOR SINGLE CLASSIFCATION,
    THE NEW USES SOFTMAX FOR MULTI CLASSIFICATION

'''

#import all libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import random
import warnings
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.arima.model import ARIMA

#======= LOAD DATASETS AND TRAIN MODELS ========

from google.colab import files
from tensorflow.keras.optimizers.schedules import CosineDecay


# Prompt user to upload files
print("Please select the file(s) to upload.")
uploaded2 = files.upload()

# Display the uploaded file names
for filename2 in uploaded2.keys():
    print(f"Uploaded file: {filename2}")

# Optional: Save the uploaded files locally in Colab
for filename2, content in uploaded2.items():
    with open(filename2, 'wb') as f:
        f.write(content)
        print(f"File saved locally as: {filename2}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sample_df = pd.read_csv(filename2)
'''
sample_df['Date'] = pd.to_datetime(sample_df.index)
sample_df.set_index('Date', inplace=True)
'''


X_synthetic = sample_df.drop(columns=['Total Spend','Average Spending Per Month','Churn State', 'TensorFlow Flag', 'Low Spending Flag'])
y_synthetic = sample_df['Churn State']
X_synthetic = X_synthetic.fillna(0)

#normalize columns
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-100, 100))

X_synthetic['Spending Trend'] = scaler.fit_transform(X_synthetic[['Spending Trend']])
X_synthetic['TensorFlow Loss'] = scaler.fit_transform(X_synthetic[['TensorFlow Loss']])


#best results for 0.15
#BEST RESULT: 0.97
X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.2, random_state=42)

# Train a Random Forest model
at_risk_model = RandomForestClassifier(n_estimators=5000, random_state=42)
at_risk_model.fit(X_train, y_train)

# Evaluate on the synthetic test data
y_pred = at_risk_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#Train TensorFlow Model

X = X_synthetic
y = y_synthetic.astype(float)


# Apply MinMaxScaler to the specified columns
#X = scaler.fit_transform(X)

X


split_index = int(len(X) * 0.75)

X_train_tf, X_test_tf = X[:split_index], X[split_index:]
y_train_tf, y_test_tf = y[:split_index], y[split_index:]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000
)


optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Define the model architecture
classification = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_tf.shape[1],)),  # Input layer
    Dropout(0.3),  # Dropout to reduce overfitting
    #Dense(756, activation='relu'),  # Hidden layer
    #Dense(512, activation = 'relu'),
    #Dense(412, activation = 'relu'),
    #Dropout(0.2),
    #Dense(356, activation = 'relu'),
    #Dense(200, activation = 'relu'),
    #Dense(196, activation='relu'),
    #Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer (0 or 1 probability)
])

# Compile the model
classification.compile(optimizer=optimizer,
              loss='binary_crossentropy',  # Binary classification loss
              metrics=['accuracy'])

# Train the model
classification.fit(X_train_tf, y_train_tf, epochs=100, batch_size=12, validation_data=(X_test_tf, y_test_tf))

# ========== DEFINE PREDICTION FUNCTION ============
def build_prediction(df, at_risk_model, sigmoid_model, customer):

  from scipy.fftpack import fft, fftfreq
  from statsmodels.tsa.arima.model import ARIMA
  import numpy as np

  warnings.filterwarnings('ignore')

  df['Date'] = pd.to_datetime(df.index)
  df.set_index('Date', inplace=True)
  weekly_sales = df.resample('W').sum()

  monthly_sales = df.resample('M').sum()
  monthly_sales

  monthly_spend_normalized = (monthly_sales - monthly_sales.min()) * (100 - 0) / (monthly_sales.max() - monthly_sales.min())
  spending_trend = np.mean(monthly_spend_normalized[-4:]) - np.mean(monthly_spend_normalized[-12:-4])

  total_spend = monthly_spend_normalized.sum().sum()
  average_monthly_spend = total_spend / 12

  # Find index of the last nonzero purchase
  import numpy as np


  # Find indices where purchases happened
  nonzero_indices = np.where(weekly_sales > 0)[0]

  # Weeks since last purchase
  if len(nonzero_indices) > 0:
      last_purchase_week = len(weekly_sales) - nonzero_indices[-1] - 1  # Corrected subtraction order
  else:
      last_purchase_week = len(weekly_sales)  # If no purchases, return full length

  # Count total zero-sale weeks
  num_zero_weeks = (weekly_sales == 0).sum()  # Corrected zero-week count

  # Output results
  print("Weeks since last purchase:", last_purchase_week)
  print("Total zero-sale weeks:", num_zero_weeks)


  total_0_weeks = 0
  weekly_data = weekly_sales.to_numpy()
  for i in range(len(weekly_data)):
    if weekly_data[i] == 0:
      total_0_weeks += 1

  print(f"total 0 weeks: {total_0_weeks}")
  print(f"num 0 weeks: {num_zero_weeks}")
  peak_amplitudes = []
  peak_frequencies = []
  quarter_length = 13

  for i in range(4):
      # Define the start and end of each quarter
      start_idx = i * quarter_length
      end_idx = start_idx + quarter_length

      # Slice the data to get the current quarter
      quarter_data = weekly_sales[start_idx:end_idx]

      # Perform Fourier Transform on the current quarter's data
      fourier_y = fft(quarter_data)
      fourier_x = fftfreq(len(quarter_data), 1.0)[:len(quarter_data) // 2]  # Frequency axis for this quarter

      # Calculate the amplitudes and peak frequency for this quarter
      amplitudes = 2.0 / len(quarter_data) * np.abs(fourier_y[:len(quarter_data) // 2])
      peak_freq = fourier_x[np.argmax(amplitudes)]  # Peak frequency
      peak_amplitude = np.max(amplitudes)  # Peak amplitude

      # Store the peak amplitude for this quarter
      peak_amplitudes.append(peak_amplitude)
      peak_frequencies.append(peak_freq)

  # Now, peak_amplitudes contains the peak amplitude for each of the four quarters

  q1_f, q2_f, q3_f, q4_f = peak_amplitudes
  q1_pf, q2_pf, q3_pf, q4_pf = peak_frequencies

  try:

    net_fourier_diff = (q4_f - q1_f) / np.mean(peak_amplitudes)

    net_freq_diff = (q4_pf - q1_pf) / np.mean(peak_amplitudes)

  except:
    net_fourier_diff = 0
    net_freq_diff = 0

  avg_fourier_diff = q4_f - np.mean(peak_amplitudes)

  avg_freq_diff = q4_pf - np.mean(peak_frequencies)

  from tensorflow.keras.layers import Dense, LSTM
  from tensorflow.keras.models import Sequential

  window_size = 13  # Use the previous quarter (13 weeks) as the input window

  X = []
  y = []

  sales_values = weekly_sales.values.astype(float)

  # Create sliding window features
  for i in range(window_size, len(sales_values)):
      X.append(sales_values[i - window_size:i])  # Last quarter's data
      y.append(sales_values[i])  # Target is this week's sales

  X = np.array(X)
  y = np.array(y)


  # Manual time-based train-test split (75% train, 25% test)
  split_index = int(len(X) * 0.75)

  X_train, X_test = X[:split_index], X[split_index:]
  y_train, y_test = y[:split_index], y[split_index:]

  # Print shapes to verify

  from sklearn.preprocessing import MinMaxScaler

  scaler = MinMaxScaler(feature_range=(0, 1))

  X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Reshape to 2D for scaling
  X_train_scaled = scaler.fit_transform(X_train_reshaped)

  X_test_reshaped = X_test.reshape(X_test.shape[0], -1)  # Reshape test data
  X_test_scaled = scaler.transform(X_test_reshaped)


  # Build a simple LSTM model for financial prediction
  model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(1)  # Output layer (1 value: predicted weekly sales)
  ])

  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(X_train_scaled, y_train, epochs=25, batch_size=32)
  loss = model.evaluate(X_test, y_test)


  weeks = []
  for i in range(len(weekly_data) - 3):
      ind = i + 3
      d = weekly_data[ind]  # True value for next week
      try:
          # Fit ARIMA model to predict the next value (week)
          model = ARIMA(weekly_data[max(0, ind-12):ind], order=(4, 0, 1))
          fitted_model = model.fit()
          ARIMA_pred = fitted_model.forecast(steps=1)[0]  # Forecast the next value (next week)
      except:
          ARIMA_pred = 1  # Default value if ARIMA fails (you may adjust this)

      try:
          # Check if the true value is 30% outside the predicted value
          if ARIMA_pred != 0 and (d < ARIMA_pred * 0.3):
              weeks.append(1)  # Mark as abnormal week
          else:
              weeks.append(0)  # Normal week
      except:
          weeks.append(0)  # Handle any errors gracefully


  # Calculate the total number of abnormal weeks
  total_abnormal_weeks = sum(weeks)

  low_spending = (total_abnormal_weeks > 13)

  tensorflow_flag = loss < 1

  data = []

  data2 = []


  data.append([last_purchase_week, total_0_weeks,
               spending_trend, total_abnormal_weeks,
              q1_f, q4_f, net_fourier_diff, avg_fourier_diff,
               loss])

  result = at_risk_model.predict(data)[0]

  result_df = pd.DataFrame(data=data)

  sigmoid_predict = sigmoid_model.predict(result_df)[0][0]

  data2 = [customer, result, sigmoid_predict, last_purchase_week, total_0_weeks,
           spending_trend, total_abnormal_weeks,
           q1_f, q4_f, net_fourier_diff, avg_fourier_diff, loss
           ]

  columns = ['Customer', 'Predicted At Risk', 'Risk Score','Weeks Since Last Purchase', 'Amount of 0 Sell Weeks',
            'Spending Trend',
           'Total Abnormal Weeks', 'Q1 Amplitude', 'Q4 Amplitude', 'Net Fourier Difference',
           'Net Fourier Difference From Average',
           'TensorFlow Loss']

  print(len(data2))
  print(len(columns))
  dataf = pd.DataFrame(data=[data2], columns=columns)



  return dataf

# ============ IMPORT CUSTOMER DATA =================

from google.colab import files

# Prompt user to upload files
print("Please select the file(s) to upload.")
uploaded2 = files.upload()

# Display the uploaded file names
for filename2 in uploaded2.keys():
    print(f"Uploaded file: {filename2}")

# Optional: Save the uploaded files locally in Colab
for filename2, content in uploaded2.items():
    with open(filename2, 'wb') as f:
        f.write(content)
        print(f"File saved locally as: {filename2}")

df = pd.read_csv(filename2)
df.fillna(0, inplace=True)
df = df[1:]
customer_list = df[' Code']
df = df.drop(columns = [' Code', ' Total Value'])
df = df.astype(float)

# ================= BULD PREDICTIONS ====================

inp = 5
df2 = df.head(100)

final_df = pd.DataFrame()

for i in range(len(df2)):
  row = df2.iloc[i]
  row = row.T
  row = row.to_frame()
  #row = row.apply(lambda x: (x - x.min()) * 100 / (x.max() - x.min()))
  customer = customer_list.iloc[i]
  print(customer)
  result = build_prediction(row, at_risk_model, classification, customer)
  final_df = pd.concat([final_df, result])


# =============== EXPORT RESULTS TO CSV ===================
final_df.to_csv('Real_Customer_Preds.csv', index=False)
files.download('Real_Customer_Preds.csv')



