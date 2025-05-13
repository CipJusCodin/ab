# Program 2: Build a Simple Sequential CNN Model for CIFAR-10

```python
import tensorflow as tf

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values
X_train = X_train / 255.
X_test = X_test / 255.

# Build the CNN model
model = tf.keras.Sequential([
    # First convolutional layer
    tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", 
                         activation="relu", input_shape=(32, 32, 3)),
    # Second convolutional layer
    tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", 
                         activation="relu"),
    # Max pooling layer
    tf.keras.layers.MaxPool2D(),
    # Flatten layer
    tf.keras.layers.Flatten(),
    # Dropout layer
    tf.keras.layers.Dropout(0.25),
    # Dense layer
    tf.keras.layers.Dense(128, activation="relu"),
    # Second dropout layer
    tf.keras.layers.Dropout(0.5),
    # Output layer
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(loss="sparse_categorical_crossentropy",
             optimizer="nadam",
             metrics=["accuracy"])

# Train the model (reduced epochs for faster execution)
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1, batch_size=64)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save("cifar10_cnn_model.h5")
```
