# Laboratory Task 2: Build a simple Sequential CNN model for classifying CIFAR-10/ MNIST dataset 
```python
import tensorflow as tf

# Load and normalize data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train, X_test = X_train/255.0, X_test/255.0

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile and train
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)          # increased epochs

# Evaluate and save
print("Loss, accuracy:", model.evaluate(X_test, y_test))    #simplified print function
model.save("cifar10_model.h5")
```
