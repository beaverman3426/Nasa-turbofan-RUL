# NASA Turbofan Remaining Useful Life (RUL) Prediction

## Overview
This repository contains a machine learning project for predicting the **Remaining Useful Life (RUL)** of turbofan engines using the NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS). A **Bidirectional Long Short-Term Memory (BiLSTM)** neural network is employed to model time-series sensor data and predict the number of operational cycles remaining before engine failure. This is critical for predictive maintenance in aerospace, where engine reliability impacts safety and efficiency.

## What is RUL and Why is it Important?
**Remaining Useful Life (RUL)** is the estimated number of operational cycles an engine can perform before failing or requiring maintenance. RUL prediction is vital for:

- **Safety**: Anticipating engine failure prevents catastrophic incidents during flight.
- **Cost Efficiency**: Scheduled maintenance based on RUL minimizes downtime and repair costs.
- **Operational Planning**: Airlines and space agencies optimize maintenance schedules and resources.
- **Space Travel**: In space missions with limited repair opportunities, accurate RUL prediction ensures propulsion system reliability, safeguarding mission success and crew safety.

## Dataset Description
The NASA CMAPSS dataset simulates turbofan engine degradation under various operating conditions. Key details include:

- **Training Data**: Sensor readings over multiple cycles until failure, with RUL labels.
- **Test Data**: Partial sequences of sensor readings, where the task is to predict RUL at the sequence's end.
- **Features**: 21 sensor measurements and 3 operational settings (e.g., altitude, Mach number, throttle resolver angle).
- **Engines**: Multiple engine units, each exhibiting unique degradation patterns.

### Sensor Measurements
The dataset includes 21 sensors monitoring engine parameters. The table below describes the sensors:

![Sensor Measurements](/Data-description-of-turbofan-engine-sensor.png)

These sensors measure physical properties like temperature, pressure, and flow rates, which degrade as the engine nears failure.

## Data Preprocessing

### Normalization
Sensor data is normalized using `StandardScaler` from scikit-learn to ensure consistent feature scales:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_train[features] = scaler.fit_transform(df_train[features])
df_test[features] = scaler.transform(df_test[features]) 
```
## Windowing

The dataset is a time-series problem, requiring sequential sensor readings to be processed. **Windowing** is used to prepare the data for the BiLSTM model.

### What is Windowing?

Windowing splits time-series data into fixed-size sequences (windows) of consecutive cycles. For example, a window of 30 cycles includes sensor readings for 30 time steps, with the RUL at the last time step as the target.

### How it Works

For each engine, sequences of length `window_size` are extracted using a sliding window approach. The input data becomes a 3D tensor of shape `(samples, window_size, features)`, and the output is the RUL for the last time step in the sequence.

### Why is it Important?

- **Captures Temporal Dependencies**: Time-series models like LSTMs require sequential input to capture temporal dependencies in the data.
- **Learn Degradation Patterns**: Windowing allows the model to learn degradation patterns over time by providing context for the sensor readings at each cycle.
- **Balance Efficiency and Accuracy**: It balances capturing long-term trends (larger windows) with computational efficiency (smaller windows), ensuring the model can generalize well without being computationally expensive.

### LSTM Choice
Long Short-Term Memory (LSTM) networks are well-suited for time-series data because they can capture long-term dependencies and handle vanishing gradient issues. In the CMAPSS dataset, engine degradation patterns evolve over many cycles, making LSTMs ideal.

### Bidirectional LSTM (BiLSTM)
A BiLSTM processes the input sequence in both forward and backward directions, allowing the model to capture context from both past and future time steps within the window. This is particularly useful for RUL prediction, where degradation trends may have subtle bidirectional patterns.

### Advantages of BiLSTM
- Improved feature extraction by considering both temporal directions.
- Better performance on complex sequences compared to unidirectional LSTMs.
- Robustness to noise and variability in sensor data.

## Architecture Details
- **Input**: 17 features (selected sensors and operational settings) for each time step in the window.
- **LSTM Layers**: Two stacked BiLSTM layers with 64 hidden units each. The first layer outputs 128 units (64 forward + 64 backward), and the second layer maintains this dimensionality.
- **Dropout**: 20% dropout after each LSTM layer prevents overfitting.
- **Fully Connected Layer**: Maps the final BiLSTM output (last time step, 128 units) to a single RUL prediction.
- **Output**: A scalar representing the predicted RUL.

## Why No Activation Functions in LSTM Layers?
In both classical LSTMs and BiLSTMs, activation functions are not explicitly added after the LSTM layers because:

- **Built-in Non-linearities**: An LSTM cell already contains multiple non-linear activations:
  - Sigmoid functions for the input, forget, and output gates.
  - Tanh function for the cell state and output.
- **Cell Output**: The LSTM output is a product of a sigmoid (output gate) and a tanh (cell state), providing sufficient non-linearity.
- **BiLSTM Behavior**: BiLSTMs extend this by processing the sequence bidirectionally, but the internal non-linearities remain the same.
- **Task Specificity**: For regression tasks like RUL prediction, the final linear layer (without activation) directly outputs the predicted value, as RUL is a continuous variable.

## Results
The BiLSTM model achieved the following performance on the test set:

- **Root Mean Squared Error (RMSE)**: The model initially achieved an RMSE of approximately 15, indicating good predictive accuracy. However, the RMSE slightly increased during later epochs before stabilizing.

### Interpretation
An RMSE of 15 means the model's RUL predictions are, on average, off by 15 cycles, which is reasonable given the dataset's complexity and variability in engine degradation.

## Observations
- The model effectively captured temporal patterns in sensor data, thanks to the BiLSTM's bidirectional processing.
- The slight increase in RMSE suggests potential overfitting or sensitivity to certain engine units with irregular degradation patterns.

## Possible Improvements
To enhance the model's performance, consider the following:

- **Feature Selection**: Use domain knowledge or feature importance analysis to select the most relevant sensors, reducing noise.
- **Window Size Tuning**: Experiment with different window sizes to balance short-term and long-term dependencies.
- **Advanced Architectures**: Incorporate attention mechanisms or transformers to focus on critical time steps in the sequence.
- **Regularization**: Increase dropout or add L2 regularization to prevent overfitting.
- **Ensemble Methods**: Combine predictions from multiple models (e.g., [BiLSTM + TCN](https://www.mdpi.com/2076-3417/15/4/1702)) for improved robustness.
- **Data Augmentation**: Apply techniques like noise injection to enhance the model's generalization to unseen engine units.
