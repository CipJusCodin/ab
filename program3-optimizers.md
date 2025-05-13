# Program 3: Experiment with Different Optimizers

```python
import tensorflow as tf
import numpy as np

# Load dataset (using winequality-red.csv as mentioned in manual page 22)
data = pd.read_csv('Data/winequality-red.csv', sep=';')
y = data['quality']
X = data.drop(['quality'], axis=1)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2017)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2017)

# Define the model creation function
def create_model(opt):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    return model

# Define optimizers to test
opts = {
    'sgd': tf.keras.optimizers.SGD(),
    'sgd-0001': tf.keras.optimizers.SGD(lr=0.0001, decay=0.00001),
    'adam': tf.keras.optimizers.Adam(),
    'adadelta': tf.keras.optimizers.Adadelta(),
    'rmsprop': tf.keras.optimizers.RMSprop(),
    'rmsprop-0001': tf.keras.optimizers.RMSprop(lr=0.0001),
    'nadam': tf.keras.optimizers.Nadam(),
    'adamax': tf.keras.optimizers.Adamax()
}

# Training parameters
batch_size = 128
n_epochs = 100  # Reduced from 1000 in manual for faster execution

# Train and compare optimizers
results = []
for opt in opts:
    # Create and compile model
    model = create_model(opt)
    model.compile(loss='mse', optimizer=opts[opt], metrics=['accuracy'])
    
    # Train model
    hist = model.fit(X_train.values, y_train, 
                    batch_size=batch_size, 
                    epochs=n_epochs,
                    validation_data=(X_val.values, y_val), 
                    verbose=0)
    
    # Find epoch with best validation accuracy
    best_epoch = np.argmax(hist.history['val_accuracy'])
    best_acc = hist.history['val_accuracy'][best_epoch]
    
    # Evaluate on test set
    score = model.evaluate(X_test.values, y_test, verbose=0)
    
    # Save results
    results.append([opt, best_epoch, best_acc, score[1]])

# Display results as dataframe
res = pd.DataFrame(results)
res.columns = ['optimizer', 'epochs', 'val_accuracy', 'test_accuracy']
print(res)
```