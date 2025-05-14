# Laboratory Task 3: Experiment with different optimizers (e.g., Adam vs. RMSProp) and compare their impact on accuracy and convergence.

```python
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and prepare data
data = pd.read_csv('Data/winequality-red.csv', sep=';')
X = data.drop(['quality'], axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create model function
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(100, input_dim=X_train.shape[1], activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

# Define optimizers
optimizers = {
    'sgd': tf.keras.optimizers.SGD(),
    'adam': tf.keras.optimizers.Adam(),
    'rmsprop': tf.keras.optimizers.RMSprop(),
    'nadam': tf.keras.optimizers.Nadam()
}

# Compare optimizers
results = []
for name, opt in optimizers.items():
    # Train model
    model = create_model()
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    hist = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.2, verbose=0)
    
    # Evaluate
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    best_val_acc = max(hist.history['val_accuracy'])
    results.append([name, best_val_acc, acc])

# Print results
df = pd.DataFrame(results, columns=['optimizer', 'val_accuracy', 'test_accuracy'])
print(df)
```
