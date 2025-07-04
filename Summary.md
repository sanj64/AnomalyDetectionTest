# Summary

## Anomaly Detection with Isolation Forest
**Goal**: To detect unusual patterns (anomalies) in synthetic time-series data using the Isolation Forest algorithm.

### How Isolation Forest Works:

Isolation Forest builds multiple "isolation trees" by randomly selecting a feature and then a random split value for that feature. This process is repeated recursively. Anomalies, being "different" from the majority, tend to be isolated closer to the root of these trees (requiring fewer splits), resulting in shorter "path lengths." Normal data points, being more densely packed, require more splits to be isolated, leading to longer path lengths. The anomaly score is based on these path lengths: shorter paths indicate higher anomaly likelihood.

### What is Anomaly Detection? 

Anomaly detection, also known as outlier detection, is the process of identifying data points that deviate significantly from the majority of the data. These "anomalies" can indicate critical events, errors, or interesting insights. In time-series data, an anomaly might be an unusually high sensor reading, a sudden drop in website traffic, or a burst of network activity.

### Why Isolation Forest? 

Isolation Forest is an unsupervised machine learning algorithm particularly well-suited for anomaly detection. Its core idea is that anomalies are "few and different" and thus easier to isolate than normal data points. It does this by randomly partitioning data and observing how many splits it takes to isolate a data point. Anomalies typically require fewer splits to be isolated.

#
This script [anomaly_detector.py](https://github.com/sanj64/AnomalyDetectionTest/blob/main/anomaly_detector.py) demonstrates a simple Anomaly Detection task using the Isolation Forest algorithm.

#### 1. Data Generation:

Synthetic time-series data is created with a gentle trend and seasonality to mimic real-world sensor data.
Specific anomalies (sudden spikes, drops, and a sustained unusual period) are deliberately injected into this data to serve as ground truth for our detection.

#### 2. Data Preparation:

The value column (representing our time-series measurement) is selected as the feature for the model.
The data is optionally scaled using StandardScaler to ensure all features contribute equally, although Isolation Forest is robust to unscaled data.

#### 3. Model Training:

An IsolationForest model from scikit-learn is initialized.
Key parameters like n_estimators (number of trees) and contamination (estimated proportion of anomalies) are set.
The model is trained on the prepared data. Being an unsupervised algorithm, it learns patterns of "normality" without needing explicit anomaly labels.

#### 4. Anomaly Prediction & Scoring:

The trained model predicts whether each data point is an anomaly (-1) or normal (1).
It also provides decision_function scores, where lower (more negative) scores indicate a higher likelihood of being an anomaly.

#### 5. Visualization:

The original time-series data is plotted, with detected anomalies highlighted in a distinct color (red circles), making the results visually intuitive.
A histogram of anomaly scores is provided to understand the distribution of anomaly likelihoods.
