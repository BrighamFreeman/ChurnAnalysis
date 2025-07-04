'''

################################
#    THIS CODE CONTAINS THE    #
#  WGAN-GP USED FOR SYNTHETIC  #
#  TIME SERIES DATA CREATION   #
################################

'''

# IMPORT LIBRARIES
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import random
import warnings
from scipy.fftpack import fft, fftfreq
from statsmodels.tsa.arima.model import ARIMA

# UPLOAD DATA TO MODEL WGAN ON
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


import pandas as pd
import numpy as np
import random

# Load and preprocess the data
df = pd.read_csv(filename2)
df.fillna(0, inplace=True)
df = df[1:]
df_sorted = df.sort_values(by=' Total Value')
df = df.head(int(len(df) * 0.33))

def generate_augmented(churned):
  rand = random.randint(0,len(df)-1)
  row = df.iloc[rand][2:]
  row.index = pd.to_datetime(row.index)
  weekly_sales = row.resample('W').sum()
  weekly_sales = weekly_sales.astype(float)
  #weekly_sales = (weekly_sales - weekly_sales.min()) / (weekly_sales.max() - weekly_sales.min()) * 100
  weeks = 52  # 12 months of weekly data
  spending = np.zeros(52)
  start_spend = np.random.uniform(0, 1000)

  if churned:
      # Simulate spending decrease over time, leading to 0
      drop_point = np.random.randint(15, 40)  # Random point where spending drops
      decay = np.random.uniform(0.6, 0.95)  # How fast spending drops
      for i in range(drop_point):
        ggg = random.random() < 0.6
        if ggg:
          spending[i] = 0
        else:
          spending[i] = weekly_sales.iloc[i] + np.random.normal(0, start_spend * 0.1)

      for i in range(drop_point, weeks):
        #introduce 0 sell weeks
        no_sell = random.random() < 0.6
        if no_sell:
          spending[i] = 0
        else:
          no_sell = random.random() < 0.77
          if no_sell:
            if i < len(weekly_sales):
                spending[i] = weekly_sales.iloc[i]
            else:
                spending[i] = 0  # Handle missing weeks gracefully

          else:
            try:
              spending[i] = spending[i-1] * (decay ** (i * 2))
            except:
              spending[i] = start_spend * decay ** (i * 2)

  else:
    for i in range(52):

      random_drop = random.random() < 0.05
      if random_drop:
        spending[i] = 0
      else:
        spending[i] = row.iloc[i] + np.random.normal(0, start_spend * 0.1)

  return spending


def untouched_data(churned):
  rand = random.randint(0,len(df)-1)
  row = df.iloc[rand][2:]
  row.index = pd.to_datetime(row.index)
  weeks = 52
  weekly_sales = row.resample('W').sum()
  weekly_sales = weekly_sales.astype(float)
  #weekly_sales = (weekly_sales - weekly_sales.min()) / (weekly_sales.max() - weekly_sales.min()) * 100
  spending = np.zeros(weeks)
  for i in range(weeks):
    spending[i] = weekly_sales.iloc[i]

  return spending

import numpy as np
import pandas as pd
import random

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 5000

# Generate random customer IDs
customer_ids = [f"CUST_{i}" for i in range(n_customers)]


#generate synthetic dataset
data = []

for customer in customer_ids:
  print(f"Generating {customer}")
  #set churn_state to true 33% of the time
  churned = random.random() < 0.33
  generator_seed = random.random()
  if generator_seed < 0.70: # 30% chance to use real customer data
    churned = False
    #set flag to false, since customer is very likely not going to be at risk
    c = untouched_data(churned)
  else:
   c = generate_augmented(churned)

  data.append(c)

data

data = np.array(data)

from sklearn.preprocessing import MinMaxScaler

# Create the scaler object
scaler = MinMaxScaler(feature_range=(-1, 1))


# Fit and transform the data to the range [-1, 1]
data = scaler.fit_transform(data)

df2 = pd.DataFrame(data)

data.shape

#=============== WGAN MODEL ========================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Gradient Penalty Function
def gradient_penalty(discriminator, real_data, fake_data):
    batch_size = real_data.shape[0]
    epsilon = tf.random.uniform([batch_size, 1, 1])
    interpolated_data = epsilon * real_data + (1 - epsilon) * fake_data

    with tf.GradientTape() as tape:
        tape.watch(interpolated_data)
        disc_output = discriminator(interpolated_data, training=True)

    gradients = tape.gradient(disc_output, interpolated_data)
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    gp = tf.reduce_mean(tf.square(gradient_norm - 1.0))
    return 10 * gp

# LSTM Generator
# Conv1D Generator
def build_generator(latent_dim, time_steps, num_features):
    model = keras.Sequential([
        layers.Dense(1536, activation="tanh", input_shape=(latent_dim,)),  # First dense layer remains
        layers.RepeatVector(time_steps),  # Repeat latent vector across time steps

        # Conv1D layers at the beginning for feature extraction
        layers.Conv1D(256, 3, activation='tanh', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(164, 5, activation='tanh', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(128, 7, activation='tanh', padding='same'),
        layers.Conv1D(64, 9, activation = 'tanh', padding = 'same'),

        layers.Dense(512),

        # LSTM layers for capturing temporal dependencies
        layers.LSTM(1024, return_sequences=True, activation='tanh'),
        layers.Dense(756, activation='swish'),
        layers.LSTM(512, return_sequences=True, activation='tanh'),
        layers.Dense(396, activation='tanh'),
        layers.Dropout(0.3),
        layers.LSTM(312, return_sequences=True, activation='tanh'),
        layers.Dense(296, activation='tanh'),
        layers.LSTM(256, return_sequences=True, activation='tanh'),
        layers.Dropout(0.2),

        # Output layer
        layers.Dense(num_features, activation='tanh')
    ])
    return model


# LSTM Discriminator
# Conv1D Discriminator
def build_discriminator(time_steps, num_features):
    model = keras.Sequential([
        layers.Conv1D(128, kernel_size=5, strides=2, padding="same", input_shape=(time_steps, num_features)),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv1D(256, kernel_size=5, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),

        layers.Conv1D(512, kernel_size=5, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),

        layers.Flatten(),
        layers.Dense(1)  # No activation, as it represents the WGAN score
    ])
    return model

#set a segment to 0 is between (-0.5,0.5)
def set_zero(x):
    x[(x < 0.5) & (x > -0.5)] = 0  # Vectorized operation
    return x

#compile generator and critic
def build_gan(data):
  # Set Parameters
  time_steps = 13
  num_features = 52
  num_generated_samples = 1000
  latent_dim = 100

  generator = build_generator(latent_dim, time_steps, num_features)
  discriminator = build_discriminator(time_steps, num_features)

  gen_optimizer = keras.optimizers.Adam(0.00004, beta_1=0.4, beta_2=0.8)
  disc_optimizer = keras.optimizers.Adam(0.00001, beta_1=0.5, beta_2=0.9)

  epochs = 50
  batch_size = 16
  critic_steps = 3

  # Simulated Data
  num_samples = len(data)

  # Reshape Data to (num_sequences, time_steps, num_features)
  num_sequences = num_samples // time_steps
  reshaped_data = data[:num_sequences * time_steps].reshape(num_sequences, time_steps, num_features)

  # Training Loop
  for epoch in range(epochs):
      for _ in range(critic_steps):
          idx = np.random.randint(0, reshaped_data.shape[0], batch_size)  # Select valid sequences
          real_samples = reshaped_data[idx]  # Correct shape: (batch_size, time_steps, num_features)

          noise = np.random.normal(-1, 1, (batch_size, latent_dim))

          num_changes = random.randint(5,15)

          # Randomly select positions without replacement
          indices = np.random.choice(len(noise), num_changes, replace=False)

          # Add 1 to the selected positions

          noise[indices] += 0.3

          fake_samples = generator(noise, training=True)

          with tf.GradientTape() as tape:
              D_real = discriminator(real_samples, training=True)
              D_fake = discriminator(fake_samples, training=True)
              gp = gradient_penalty(discriminator, real_samples, fake_samples)
              d_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + gp

          gradients = tape.gradient(d_loss, discriminator.trainable_variables)
          disc_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

      noise = np.random.normal(-1, 1, (batch_size, latent_dim))
      with tf.GradientTape() as tape:
          fake_samples = generator(noise, training=True)
          D_fake = discriminator(fake_samples, training=True)
          g_loss = -tf.reduce_mean(D_fake)

      gradients = tape.gradient(g_loss, generator.trainable_variables)
      gen_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))

      if epoch % 5 == 0:
          print(f"Epoch {epoch} | D Loss: {d_loss.numpy():.4f} | G Loss: {g_loss.numpy():.4f}")

  # Generate New Sequences
  noise = np.random.normal(-1, 1, (num_generated_samples, latent_dim))
  random_data = np.random.uniform(-0.2, 0.7, (num_generated_samples, latent_dim))

  noise += random_data
  generated_data = generator.predict(noise)

  scales = np.array([1, 10, 20, 50, 100, 250, 500, 1000])  # Define scales once


  new_data = []


  for _ in range(1):

    num_generated_samples = 1000

    noise = np.random.normal(-1, 1, (num_generated_samples, latent_dim))
    random_data = np.random.uniform(0, 0.5, (num_generated_samples, latent_dim))

    noise += random_data
    generated_data = generator.predict(noise)

    scales = np.array([1, 10, 5, 10, 100, 250, 500])  # Define scales once

    for i in range(len(generated_data)):
        scale = random.choice(scales)
        generated_data[i] = set_zero(generated_data[i])  # Apply set_zero
        generated_data[i] *= scale  # Scale the data
        generated_data[i] = abs(generated_data[i])

    cc = random.choice(generated_data)  # Randomly pick one generated sample
    new_data.append(cc)

  print(new_data[0])

  drop_columns = len(new_data[0]) - random.randint(0,20)

  new_data[0] = new_data[0][drop_columns:]

  result = np.sum(new_data[0], axis=0).reshape(1,-1)

  return result

# CREATE SYNTHETIC TIME SERIES

import numpy as np
import random


npdata = build_gan(data)

for _ in range(20):

  print(_)
  npdata = np.vstack([npdata, build_gan(data)])

# ============ GENERATE AND DOWNLOAD SYNTHETIC DATA TO BE FED INTO CHURN ANALYSIS ===============

data = []

generated_data = npdata

for i in range(len(generated_data)):

  c = generated_data[i]

  last_purchase_week = np.argmax(c[::-1] > 0)
  total_spend = np.sum(c)
  average_monthly_spend = total_spend / 12
  spending_trend = np.mean(c[-12:]) - np.mean(c[:12])  # First quarter vs last quarter spending

  total_0_weeks = 0
  for i in range(len(c)):
    if c[i] == 0:
      total_0_weeks += 1

  weeks = []
  for i in range(len(c) - 3):
      ind = i + 3
      d = c[ind]  # True value for next week
      try:
          # Fit ARIMA model to predict the next value (week)
          from statsmodels.tsa.arima.model import ARIMA
          import warnings

          # Suppress warnings
          warnings.filterwarnings("ignore")
          model = ARIMA(c[max(0, ind-12):ind], order=(4, 0, 1))
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

  """ Fourier Analysis """

  N = len(c)

  # Number of weeks in each quarter (since we're dealing with 52 weeks of data, each quarter is 13 weeks)
  quarter_length = 13

  # Initialize list to store peak amplitudes for each quarter
  peak_amplitudes = []
  peak_frequencies = []

  for i in range(4):
      # Define the start and end of each quarter
      start_idx = i * quarter_length
      end_idx = start_idx + quarter_length

      # Slice the data to get the current quarter
      quarter_data = c[start_idx:end_idx]

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

  #scaling based on average
  try:

    net_fourier_diff = (q4_f - q1_f) / np.mean(peak_amplitudes)

    net_freq_diff = (q4_pf - q1_pf) / np.mean(peak_amplitudes)

  except:
    net_fourier_diff = 0
    net_freq_diff = 0

  avg_fourier_diff = q4_f - np.mean(peak_amplitudes)

  avg_freq_diff = q4_pf - np.mean(peak_frequencies)

  """ develop tensorflow flag system """

  from tensorflow.keras.layers import Dense, LSTM
  from tensorflow.keras.models import Sequential

  window_size = 13  # Use the previous quarter (13 weeks) as the input window

  X = []
  y = []

  # Create sliding window features
  for i in range(window_size, len(c)):
      X.append(c[i - window_size:i])  # Last quarter's data
      y.append(c[i])  # Target is this week's sales

  X = np.array(X)
  y = np.array(y)

  # Manual time-based train-test split (75% train, 25% test)
  split_index = int(len(X) * 0.75)

  X_train, X_test = X[:split_index], X[split_index:]
  y_train, y_test = y[:split_index], y[split_index:]

  # Print shapes to verify
  """
  print("X_train shape:", X_train.shape)
  print("y_train shape:", y_train.shape)
  print("X_test shape:", X_test.shape)
  print("y_test shape:", y_test.shape)

  """
  from sklearn.preprocessing import MinMaxScaler

  scaler = MinMaxScaler(feature_range=(0, 1))

  # Fit the scaler to your training data and transform it
  X_train_scaled = scaler.fit_transform(X_train)

  # Apply the same transformation to your test data
  X_test_scaled = scaler.transform(X_test)

  # Check the results (scaled data between 0 and 1)
  print(X_train_scaled)
  print(X_test_scaled)


  # Build a simple LSTM model for financial prediction
  model = Sequential([
    Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(1)  # Output layer (1 value: predicted weekly sales)
  ])

  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(X_train_scaled, y_train, epochs=5, batch_size=32)
  loss = model.evaluate(X_test, y_test)

  tensorflow_flag = (loss < 1)

  low_spending = (total_abnormal_weeks > 13)

  #=================== NEW LOGIC FOR SETTING STATE =====================
  state = "NEUTRAL"

  # get scaled spending

  scaled_av_spend = spending_trend / total_spend
  print(f"Scaled spending: {scaled_av_spend}")

  if total_0_weeks > 30:
    state = "AT RISK"

  if net_fourier_diff < -1:
    if spending_trend < -20:
      state = "AT RISK"

  if net_fourier_diff > 1.5:
    if spending_trend > 50:
      state = "OVERPERFORMING"

  print(churned)

  data.append([customer, total_spend, last_purchase_week, total_0_weeks,
               average_monthly_spend, spending_trend, total_abnormal_weeks,
              q1_f, q4_f, net_fourier_diff, avg_fourier_diff,
               loss, tensorflow_flag, low_spending, state])

columns = ['Customer ID', 'Total Spend', 'Weeks Since Last Purchase', 'Amount of 0 Sell Weeks','Average Spending Per Month','Spending Trend',
           'Total Abnormal Weeks', 'Q1 Amplitude', 'Q4 Amplitude', 'Net Fourier Difference',
           'Net Fourier Difference From Average',
           'TensorFlow Loss','TensorFlow Flag','Low Spending Flag','Churn State']

at_risk_df = pd.DataFrame(data=data, columns=columns)

# DOWNLOAD THE CSV FILE
from google.colab import files
at_risk_df.to_csv('synthetic_risk_data.csv', index=False)
files.download('synthetic_risk_data.csv')
