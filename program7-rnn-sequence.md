# Program 7: Basic RNN for Sequence Prediction

```python
import tensorflow as tf
import numpy as np

# Generate sine wave and prepare data
sequence = np.sin(np.linspace(0, 50, 100))

# Create input-output pairs (X: 10 timesteps, y: next value)
X, y = [], []
for i in range(90):  # 100 - 10
    X.append(sequence[i:i+10])
    y.append(sequence[i+10])
X, y = np.array(X).reshape(-1, 10, 1), np.array(y)

# Build, compile and train minimal RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(10, input_shape=(10, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=1)

# Print sample predictions
print("Expected:", y[:5])
print("Predicted:", model.predict(X[:5]).flatten())
```
