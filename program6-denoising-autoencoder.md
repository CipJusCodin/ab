# Laboratory Task 6: Create a denoising autoencoder to remove noise from images.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, _), (X_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize, add noise, and reshape in fewer steps
X_train = X_train / 255.0
X_test = X_test / 255.0

noise_factor = 0.3
X_train_noisy = np.clip(X_train + noise_factor * np.random.normal(size=X_train.shape), 0., 1.)
X_test_noisy = np.clip(X_test + noise_factor * np.random.normal(size=X_test.shape), 0., 1.)

# Reshape for CNN (add channel dimension)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train_noisy = X_train_noisy.reshape(-1, 28, 28, 1)
X_test_noisy = X_test_noisy.reshape(-1, 28, 28, 1)

# Build autoencoder model
autoencoder = tf.keras.Sequential([
    # Encoder
    tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same', activation='relu'),
    
    # Decoder
    tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')
])

# Compile and train model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(X_train_noisy, X_train, epochs=5, batch_size=128, validation_data=(X_test_noisy, X_test))

# Generate denoised images
denoised_images = autoencoder.predict(X_test_noisy[:10])

fig, axes = plt.subplots(2, 10, figsize=(20, 4))
for i in range(10):
    axes[0, i].imshow(X_test_noisy[i].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[1, i].imshow(denoised_images[i].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    
plt.show()
```
