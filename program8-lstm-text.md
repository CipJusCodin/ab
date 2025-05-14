# Program 8: LSTM for Text Generation

```python
import tensorflow as tf
import numpy as np

# Load and prepare data
text = open("shakespeare.txt", "r").read().lower()
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Create training data
seq_length = 100
X, y = [], []
for i in range(len(text) - seq_length):
    X.append([char_to_idx[c] for c in text[i:i+seq_length]])
    y.append(char_to_idx[text[i+seq_length]])

X = np.array(X) / len(chars)
y = tf.keras.utils.to_categorical(y, num_classes=len(chars))
X = X.reshape(-1, seq_length, 1)

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(seq_length, 1), return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.fit(X, y, batch_size=128, epochs=10)

# Text generation function
def generate_text(seed, length=100):
    result = seed
    for _ in range(length):
        x = np.array([char_to_idx[c] for c in result[-seq_length:]])
        x = (x / len(chars)).reshape(1, seq_length, 1)
        next_char = idx_to_char[np.argmax(model.predict(x, verbose=0)[0])]
        result += next_char
    return result

print(generate_text("shall i compare thee to ", 200))
```
