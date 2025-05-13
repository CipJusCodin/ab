# Program 4: Fine-tune a Pretrained Model (ResNet50)

```python
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Data directories
train_dir = "path/to/dataset/train"
val_dir = "path/to/dataset/val"

# Data preparation
datagen = ImageDataGenerator(rescale=1.0/255)
train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32)
val_gen = datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32)
num_classes = len(train_gen.class_indices)

# Load base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

# Initial training
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True
model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=3)

# Save model
model.save("fine_tuned_model.h5")
```