# Lab Task 1: Basic Tensor Operations in TensorFlow

## Core Concept
This lab focuses on performing fundamental operations on tensors using TensorFlow, like addition, multiplication, reshaping, and other mathematical operations.

## Key Operations Covered
1. Tensor creation
2. Element-wise addition
3. Element-wise subtraction
4. Element-wise multiplication
5. Element-wise division (with handling for division by zero)
6. Tensor reshaping
7. Tensor squaring
8. Broadcasting operations
9. Combining tensors (concatenation)
10. Advanced element-wise operations (max, min, abs, log, exp)

## Simplest Version of Program 1

```python
import tensorflow as tf
import numpy as np

# 1. Create a tensor
tensor = tf.constant([100, 200, 300])
print("Tensor shape:", tensor.shape)
print("Data type:", tensor.dtype)
```

**Output:**
```
Tensor shape: (3,)
Data type: <dtype: 'int32'>
```

```python
# 2. Element-wise addition
a = tf.constant([10, 20, 30])
b = tf.constant([5, 15, 25])
addition = tf.add(a, b)  # or use a + b
print("Addition result:", addition.numpy())
```

**Output:**
```
Addition result: [15 35 55]
```

```python
# 3. Element-wise subtraction
subtraction = tf.subtract(a, b)  # or use a - b
print("Subtraction result:", subtraction.numpy())
```

**Output:**
```
Subtraction result: [ 5  5  5]
```

```python
# 4. Element-wise multiplication
c = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
d = tf.constant([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
multiplication = tf.multiply(c, d)  # or use c * d
print("Multiplication result:\n", multiplication.numpy())
```

**Output:**
```
Multiplication result:
[[ 9 16 21]
 [24 25 24]
 [21 16  9]]
```

```python
# 5. Element-wise division (safe)
e = tf.constant([6.0, 8.0, 12.0, 15.0])
f = tf.constant([2.0, 3.0, 4.0, 0.0])  # Note: contains zero
safe_division = tf.where(f != 0, tf.divide(e, f), tf.zeros_like(e))
print("Safe division result:", safe_division.numpy())
```

**Output:**
```
Safe division result: [3.         2.6666667  3.         0.        ]
```

```python
# 6. Tensor reshaping
original = tf.constant([1, 2, 3, 4])
reshaped = tf.reshape(original, (2, 2))
print("Original tensor:", original.numpy())
print("Reshaped tensor:\n", reshaped.numpy())
```

**Output:**
```
Original tensor: [1 2 3 4]
Reshaped tensor:
[[1 2]
 [3 4]]
```

```python
# 7. Tensor square
g = tf.constant([-5, -7, 2, 5, 7], dtype=tf.float32)
squared = tf.square(g)
print("Original values:", g.numpy())
print("Squared values:", squared.numpy())
```

**Output:**
```
Original values: [-5. -7.  2.  5.  7.]
Squared values: [25. 49.  4. 25. 49.]
```

```python
# 8. Broadcasting
h = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
broadcasted = h + 5
print("Broadcasting result:\n", broadcasted.numpy())
```

**Output:**
```
Broadcasting result:
[[ 6  7  8]
 [ 9 10 11]
 [12 13 14]]
```

```python
# 9. Tensor concatenation
t1 = tf.constant([[1, 2], [3, 4]])
t2 = tf.constant([[5, 6], [7, 8]])
concatenated = tf.concat([t1, t2], axis=0)  # axis=0 for vertical stacking
print("Concatenated tensor:\n", concatenated.numpy())
```

**Output:**
```
Concatenated tensor:
[[1 2]
 [3 4]
 [5 6]
 [7 8]]
```

```python
# 10. Advanced operations
tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[4, 3], [2, 1]])
# Maximum
max_tensor = tf.maximum(tensor1, tensor2)
print("Maximum values:\n", max_tensor.numpy())
# Minimum 
min_tensor = tf.minimum(tensor1, tensor2)
print("Minimum values:\n", min_tensor.numpy())
# Absolute values
abs_tensor = tf.abs(tf.constant([[-1, -2], [3, -4]]))
print("Absolute values:\n", abs_tensor.numpy())
# Logarithm
log_tensor = tf.math.log(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print("Logarithm values:\n", log_tensor.numpy())
# Exponential
exp_tensor = tf.exp(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print("Exponential values:\n", exp_tensor.numpy())
```

**Output:**
```
Maximum values:
[[4 3]
 [3 4]]
Minimum values:
[[1 2]
 [2 1]]
Absolute values:
[[1 2]
 [3 4]]
Logarithm values:
[[0.        0.6931472]
 [1.0986123 1.3862944]]
Exponential values:
[[ 2.7182817  7.389056 ]
 [20.085537  54.59815  ]]
```

## Important Points to Remember

1. **Tensor Creation**: Use `tf.constant()` to create tensors with specific values
2. **Data Types**: TensorFlow operations work on tensors with compatible data types
3. **Element-wise Operations**: Most basic operations (+, -, *, /) work element-wise on tensors
4. **Safe Division**: Use `tf.where()` to handle division by zero
5. **Reshaping**: Use `tf.reshape()` to change the dimensions of a tensor
6. **Broadcasting**: Automatically expands smaller tensors to work with larger ones
7. **Concatenation**: Combines tensors along a specified axis
8. **Advanced Functions**: TensorFlow provides functions for more complex operations (max, min, abs, log, exp)
