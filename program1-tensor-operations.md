# Program 1: Basic Tensor Operations

```python
import tensorflow as tf
import numpy as np

# Core tensor operations
a = tf.constant([10, 20, 30], dtype=tf.float32)
b = tf.constant([5, 15, 25], dtype=tf.float32)

# Create basic tensors
t1 = tf.constant([[1, 2], [3, 4]])
t2 = tf.constant([[5, 6], [7, 8]])
tensor_a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor_b = tf.constant([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

# Basic operations
add_result = a + b
sub_result = a - b
mul_result = tf.multiply(tensor_a, tensor_b)
div_result = tf.divide(a, b)

# Safe division
safe_div = tf.where(b != 0, a / b, tf.zeros_like(a))

# Tensor manipulation
reshaped = tf.reshape(tf.range(4), (2, 2))
squared = tf.square(tf.constant([-5, 2, 7], dtype=tf.float32))
broadcast = tensor_a + 5
concat = tf.concat([t1, t2], axis=0)

# Advanced operations
max_values = tf.maximum(t1, tf.constant([[4, 3], [2, 1]]))
min_values = tf.minimum(t1, tf.constant([[4, 3], [2, 1]]))
abs_values = tf.abs(tf.constant([[-1, -2], [3, -4]]))
log_vals = tf.math.log(tf.constant([[1., 2.], [3., 4.]]))
exp_vals = tf.exp(tf.constant([[1., 2.], [3., 4.]]))

# Print results
print("Add:", add_result.numpy())
print("Subtract:", sub_result.numpy())
print("Multiply:", mul_result.numpy())
print("Safe Division:", safe_div.numpy())
print("Reshape:", reshaped.numpy())
print("Square:", squared.numpy())
print("Broadcast:", broadcast.numpy()[:2])
print("Concat:", concat.numpy())
print("Max:", max_values.numpy())
print("Min:", min_values.numpy())
print("Abs:", abs_values.numpy())
print("Log:", log_vals.numpy())
print("Exp:", exp_vals.numpy())
```
