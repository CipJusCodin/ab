# Program 1: Perform Basic Tensor Operations Using TensorFlow

```python
import tensorflow as tf
import numpy as np

# 1. Tensor Creation
tensor = tf.constant([100, 200, 300])

# 2. Element-wise Addition
ts1 = tf.constant(np.random.rand(2, 3))
ts2 = tf.constant(np.random.rand(2, 3))
result_tensor = tf.add(ts1, ts2)

# 3. Element-wise Subtraction
a = tf.constant([10, 20, 30], dtype=tf.float32)
b = tf.constant([5, 15, 25], dtype=tf.float32)
result = tf.math.subtract(a, b)

# 4. Element-wise Multiplication
tensor_a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
tensor_b = tf.constant([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
result = tf.multiply(tensor_a, tensor_b)

# 5. Element-wise Division
tensor1 = tf.constant([6, 8, 12, 15], dtype=tf.float32)
tensor2 = tf.constant([2, 3, 4, 0], dtype=tf.float32)
result = tf.where(tensor2 != 0, tf.divide(tensor1, tensor2), tf.zeros_like(tensor1))

# 6. Tensor Reshaping
initial_tensor = tf.constant([1, 2, 3, 4])
reshaped_tensor = tf.reshape(initial_tensor, (2, 2))

# 7. Tensor Square
a = tf.constant([-5, -7, 2, 5, 7], dtype=tf.float64)
res = tf.math.square(a)

# 8. Broadcasting Operations
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
result = tensor + 5

# 9. Combining Tensors
t1 = tf.constant([[1, 2], [3, 4]])
t2 = tf.constant([[5, 6], [7, 8]])
result_axis_0 = tf.concat([t1, t2], axis=0)

# 10. Advanced Element-wise Operations
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[4, 3], [2, 1]])
max_tensor = tf.maximum(tensor_a, tensor_b)
min_tensor = tf.minimum(tensor_a, tensor_b)
tensor_c = tf.constant([[-1, -2], [3, -4]])
abs_tensor = tf.abs(tensor_c)
tensor_d = tf.constant([[1.0, 2.0], [3.0, 4.0]])
log_tensor = tf.math.log(tensor_d)
exp_tensor = tf.exp(tensor_d)

# Print final results
print("Program 1 - Basic Tensor Operations")
print("-----------------------------------")
print("Element-wise Addition Result:")
print(result_tensor.numpy())
print("\nElement-wise Subtraction Result:")
print(result.numpy())
print("\nElement-wise Multiplication Result:")
print(tf.multiply(tensor_a, tensor_b).numpy())
print("\nSafe Division Result:")
print(tf.where(tensor2 != 0, tf.divide(tensor1, tensor2), tf.zeros_like(tensor1)).numpy())
print("\nReshaped Tensor:")
print(reshaped_tensor.numpy())
print("\nSquared Values:")
print(res.numpy())
print("\nBroadcasting Result:")
print(result.numpy())
print("\nConcatenated Tensor:")
print(result_axis_0.numpy())
print("\nMaximum Values:")
print(max_tensor.numpy())
print("\nMinimum Values:")
print(min_tensor.numpy())
print("\nAbsolute Values:")
print(abs_tensor.numpy())
print("\nLogarithm Values:")
print(log_tensor.numpy())
print("\nExponential Values:")
print(exp_tensor.numpy())
```
