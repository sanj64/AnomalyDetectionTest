# 1. Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import random # For injecting anomalies

# 2. Generate Synthetic Time-Series Data

# Set a random seed for reproducibility
np.random.seed(42)
random.seed(42)

# --- Generate Normal Data ---
# Number of data points
n_samples = 1000

# Create a time index
dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')

# Simulate a "normal" sensor reading with a slight trend and seasonality
# Base value with some daily fluctuation
normal_data = np.sin(np.linspace(0, 50, n_samples)) * 5 + np.linspace(0, 20, n_samples) + np.random.normal(0, 1, n_samples)

# Create a DataFrame
df = pd.DataFrame({'timestamp': dates, 'value': normal_data})

# --- Inject Anomalies ---
# Inject some sudden spikes (point anomalies)
num_spikes = 10
spike_indices = random.sample(range(n_samples), num_spikes)
for idx in spike_indices:
    df.loc[idx, 'value'] += np.random.uniform(20, 50) # Add a large random value

# Inject some sudden drops (point anomalies)
num_drops = 5
drop_indices = random.sample(range(n_samples), num_drops)
for idx in drop_indices:
    df.loc[idx, 'value'] -= np.random.uniform(20, 50) # Subtract a large random value

# Inject a short "contextual anomaly" (a sustained unusual period)
contextual_start = 500
contextual_end = 530
df.loc[contextual_start:contextual_end, 'value'] = np.random.normal(70, 5, contextual_end - contextual_start + 1)

print("Synthetic data generated with normal patterns and injected anomalies.")
print(df.head())
print("\nDataFrame Info:")
df.info()

# 3. Visualize the Raw Time-Series Data

plt.figure(figsize=(15, 7))
plt.plot(df['timestamp'], df['value'], label='Sensor Value', alpha=0.8)
plt.title('Synthetic Time-Series Data with Injected Anomalies')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.show()

print("Notice the sudden spikes, drops, and the sustained unusual period. These are our injected anomalies.")

# 4. Prepare Data for Isolation Forest

# Isolation Forest works best on numerical features.
# We will focus on the 'value' column for anomaly detection.
# If you had multiple features (e.g., temperature, pressure, humidity),
# you would select all of them here.
data_for_model = df[['value']]

# Optional: Scale the data. While Isolation Forest is not highly sensitive to scaling,
# it can sometimes help with performance and is generally good practice.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_for_model)

print(f"Original data shape: {data_for_model.shape}")
print(f"Scaled data shape: {scaled_data.shape}")
print("Data prepared for Isolation Forest.")

# 5. Train the Isolation Forest Model

# Initialize the Isolation Forest model
# key parameters:
# n_estimators: The number of trees in the forest. More trees generally lead to more robust results.
# contamination: The expected proportion of outliers in the data. This is an important parameter
#                as it defines the threshold for anomaly scores. If 'auto', the threshold is
#                determined by the original paper. We're setting it explicitly based on our
#                injected anomalies (10 spikes + 5 drops + 30 contextual points = ~45/1000 = 0.045)
#                Let's use 0.05 (5%) as a reasonable estimate.
# random_state: For reproducibility of the model's training process.
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

# Fit the model to the scaled data
# The fit method 'learns' the normal behavior from the data.
model.fit(scaled_data)

print("Isolation Forest model trained successfully.")

# 6. Predict Anomalies and Get Anomaly Scores

# `predict` method returns -1 for anomalies and 1 for normal points
df['anomaly_prediction'] = model.predict(scaled_data)

# `decision_function` returns the anomaly scores.
# Lower scores indicate a higher likelihood of being an anomaly.
df['anomaly_score'] = model.decision_function(scaled_data)

# Convert predictions to a more intuitive label
# -1 -> Anomaly, 1 -> Normal
df['is_anomaly'] = df['anomaly_prediction'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

print("Anomaly predictions and scores generated.")
print(df.head())
print("\nAnomaly counts:")
print(df['is_anomaly'].value_counts())

# 7. Visualize Detected Anomalies

plt.figure(figsize=(18, 8))

# Plot normal data points
normal_points = df[df['is_anomaly'] == 'Normal']
plt.plot(normal_points['timestamp'], normal_points['value'], 'b.', markersize=8, label='Normal Data', alpha=0.6)

# Plot anomalies
anomaly_points = df[df['is_anomaly'] == 'Anomaly']
plt.plot(anomaly_points['timestamp'], anomaly_points['value'], 'ro', markersize=6, label='Detected Anomaly', alpha=0.9)

plt.title('Time-Series Anomaly Detection with Isolation Forest')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("The red circles indicate data points detected as anomalies.")

# 8. Explore Anomaly Scores 

plt.figure(figsize=(15, 6))
plt.hist(df['anomaly_score'], bins=50, density=True, alpha=0.7, color='c')
plt.title('Distribution of Anomaly Scores')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# You can also look at the most anomalous points
print("\nTop 10 most anomalous points (lowest scores):")
print(df.sort_values(by='anomaly_score').head(10))

print("\nInterpretation of scores: Lower (more negative) scores indicate higher likelihood of being an anomaly.")
print("The 'contamination' parameter in IsolationForest sets the threshold internally.")