# Program 5: Transfer Learning with MobileNet

```python
import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist

# Preprocess data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension and resize to 224x224 for MobileNet
X_train = tf.image.resize(tf.expand_dims(X_train, -1), [224, 224])
X_test = tf.image.resize(tf.expand_dims(X_test, -1), [224, 224])

# Convert to RGB (MobileNet expects 3 channels)
X_train = tf.concat([X_train, X_train, X_train], axis=-1)
X_test = tf.concat([X_test, X_test, X_test], axis=-1)

# Create one-hot encoded labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Load pretrained MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=64, epochs=5, validation_split=0.2)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model
model.save("mobilenet_fashion_mnist.h5")
```