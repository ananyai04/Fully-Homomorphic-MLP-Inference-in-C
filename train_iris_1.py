import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# One-hot encode
Y = np.zeros((y.size, 3))
Y[np.arange(y.size), y] = 1

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# SMALL weight initialization (IMPORTANT)
np.random.seed(0)
W1 = np.random.randn(4, 4) * 0.1
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 3) * 0.1
b2 = np.zeros((1, 3))

# Square activation with clipping
def square(x):
    x = np.clip(x, -5, 5)   # prevents explosion
    return x * x

# Training
lr = 0.001   # smaller learning rate
for epoch in range(5000):

    # Forward
    Z1 = X_train @ W1 + b1
    A1 = square(Z1)
    Z2 = A1 @ W2 + b2

    loss = np.mean((Z2 - Y_train) ** 2)

    # Backprop
    dZ2 = 2 * (Z2 - Y_train) / Y_train.shape[0]
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (2 * np.clip(Z1, -5, 5))  # derivative of clipped square
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 1000 == 0:
        print("Loss:", loss)

# Save weights
np.savetxt("w1.txt", W1.flatten())
np.savetxt("b1.txt", b1.flatten())
np.savetxt("w2.txt", W2.flatten())
np.savetxt("b2.txt", b2.flatten())

print("Training complete + stable weights saved!")