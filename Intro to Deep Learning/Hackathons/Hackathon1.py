import numpy as np
import tensorflow as tf

# Equation used for all of the examples is same and is - (A * x + b)

# 1. Example 1 - we are using a learning rate of 0.15 and num_iterations = 25
learning_rate = 0.15
num_iterations = 25

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')


# 2. Example 2 - we are using a learning rate of 0.15 and num_iterations = 100

learning_rate = 0.15
num_iterations = 100

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')

# 3. Example 3 - we are using the same learning rate of 0.15 num_iterations = 250 and we see that sq. error decreases to a greater extent as compare to Example 1

learning_rate = 0.15
num_iterations = 250

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')

# 4. Example 4 - we are using the same learning rate of 0.15, num_iterations = 500 as compare to Example 1. We see a similar trend of decreasing error if we increase num_iterations

learning_rate = 0.15
num_iterations = 500

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')

# 5. Example 5 - we are using a learning rate of 0.05 and num_iterations = 50

learning_rate = 0.05
num_iterations = 50

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')

# 6. Example 6 - we are using a learning rate of 0.1 and same num_iterations = 50

learning_rate = 0.1
num_iterations = 50

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')

# 7. Example 7 - We are using a learning rate of 1.0 and same num_iterations = 50

learning_rate = 1.0
num_iterations = 50

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b

print('--------------------$$$$$$$$$$$$$$$$--------------------')

# 8. Example 8 - We are using a learning rate of 10.0 and same num_iterations = 50

learning_rate = 10.0
num_iterations = 50

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
# A = tf.random.normal([3, 4])
A = tf.random.uniform([3,4], minval=-3, maxval=3)
# Create x using an arbitrary initial value
x = tf.Variable(tf.random.normal([4, 1]))
# Create a fixed vector b
b = tf.random.uniform([3,1], minval=-3, maxval=3)
# b = tf.random.normal([3, 1])

# Check the initial values
# print("A:", A.numpy())
# print("b:", b.numpy())

# print("Initial x:", x.numpy())
# print("Ax:", (A @ x).numpy())
# print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    # print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product + b)
        z = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        # print("Squared error:", z)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        # print("Gradients:")
        # print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        # print()

# Check the final values
print("Squared error:", z)
# print("Optimized x", x.numpy())
# print("Ax", (A @ x).numpy())  # Should be close to the value of b
