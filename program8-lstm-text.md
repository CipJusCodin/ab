# Program 8: LSTM for Text Generation

```python
import tensorflow as tf
import numpy as np

# Load text data (example using a small text file)
text = open("shakespeare.txt", "r").read().lower()
chars = sorted(set(text))

# Create character to index mapping
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Prepare training data
seq_length = 100
sequences = []
next_chars = []

for i in range(len(text) - seq_length):
    sequences.append([char_to_idx[c] for c in text[i:i+seq_length]])
    next_chars.append(char_to_idx[text[i+seq_length]])

# Convert to numpy arrays and normalize
X = np.array(sequences) / len(chars)
y = tf.keras.utils.to_categorical(next_chars, num_classes=len(chars))

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, input_shape=(seq_length, 1), return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

# Reshape X to include feature dimension
X = X.reshape(X.shape[0], X.shape[1], 1)

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam")

# Train model
model.fit(X, y, batch_size=128, epochs=20)

# Function to generate text
def generate_text(seed_text, length=200):
    generated = seed_text
    
    for _ in range(length):
        # Convert seed text to sequence of indices
        x_pred = np.array([char_to_idx[c] for c in generated[-seq_length:]])
        x_pred = x_pred / len(chars)
        x_pred = x_pred.reshape(1, seq_length, 1)
        
        # Predict next character
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = idx_to_char[next_index]
        
        # Add predicted character to generated text
        generated += next_char
    
    return generated

# Generate new text
print(generate_text("shall i compare thee to a summer's day? ", length=200))
```