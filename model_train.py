import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Flatten
import numpy as np

from test import Setting

# Example usage
input_shape = (50, 1)  # Assuming time-series data with 50 timesteps and 1 feature
num_classes = 10
model = Setting.create_cnn_gru_model(input_shape, num_classes)

# Print the model summary
print(model.summary())


# Simulated dataset
X_train = np.random.rand(100, 50, 1)  # Example data: 100 samples, 50 timesteps, 1 feature
y_train = np.random.randint(0, num_classes, 100)  # Example labels: 100 samples

model.fit(X_train, y_train, epochs=1000, batch_size=32)