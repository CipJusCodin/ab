# Laboratory Task 1: Perform basic tensor operations (like addition, multiplication) using Tensor Flow. 

```python
import tensorflow as tf
import numpy as np

# Create basic tensors
a = tf.constant([10, 20, 30], dtype=tf.float32)
b = tf.constant([5, 15, 25], dtype=tf.float32)
t1 = tf.constant([[1, 2], [3, 4]])
t2 = tf.constant([[5, 6], [7, 8]])
tensor_a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor_b = tf.constant([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# Basic operations
add = a + b
sub = a - b
mul = tf.multiply(tensor_a, tensor_b)
div = tf.divide(a, b)

# Safe division
safe_div = tf.where(b != 0, a / b, tf.zeros_like(a))

# Tensor manipulation
reshaped = tf.reshape(t1, (4,))
squared = tf.square(a)
broadcast = tensor_a + 5
concat = tf.concat([t1, t2], axis=0)

# Advanced operations
max = tf.maximum(t1, t2)
min = tf.minimum(t1, t2)
abs = tf.abs(t1 - t2)
log = tf.math.log(a)
exp = tf.exp(b)

# Print results
print("Add:", add.numpy())
print("Subtract:", sub.numpy())
print("Multiply:", mul.numpy())
print("Safe Division:", safe.numpy())
print("Reshape:", reshaped.numpy())
print("Square:", squared.numpy())
print("Broadcast:", broadcast.numpy())
print("Concat:", concat.numpy())
print("Max:", max.numpy())
print("Min:", min.numpy())
print("Abs:", abs.numpy())
print("Log:", log.numpy())
print("Exp:", exp.numpy())
```
