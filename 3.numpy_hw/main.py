import numpy as np 

# Normal Distribution
def normal_distribution(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")

    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2

    return coefficient * np.exp(exponent)

result = normal_distribution(x=0.0, mu=0.0, sigma=1.0)
print(result)


# Sigmoid Function
def sigmoid_function(x: float) -> float:
    return 1 / (1 + np.exp(-x))
result_sigmoid = sigmoid_function(x=0.0)
print(result_sigmoid)
# if the input is negative, the output will be between 0 and 0.5, if the input is positive, the output will be between 0.5 and 1, and if the input is 0, the output will be 0.5.



# Weights update in Logistic Regression
def update_weights(
    w: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
    alpha: float = 0.0005,
) -> np.ndarray:
    """Update the weight vector using prediction errors."""

    n = X.shape[0]
    updated_w = w.copy()

    for i in range(n):
        error = y_hat[i] - y[i]
        updated_w = updated_w - alpha * (1 / n) * error * X[i]

    return updated_w
# Example data
w = np.array([0.5, 0.2])

X = np.array([
    [1.0, 2.0],
    [3.0, 4.0],
])

y = np.array([0.0, 1.0])
y_hat = np.array([0.4, 0.8])

# Call the function
new_w = update_weights(w, X, y, y_hat)

print("Original weights:", w)
print("Updated weights:", new_w)




# Mean Squared Error
def mean_squared_error(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> float:
    squared_errors = (y - y_hat) ** 2
    return float(np.mean(squared_errors))


# Example data
y = np.array([3.0, 5.0, 2.0, 7.0])
y_hat = np.array([2.5, 5.5, 2.0, 8.0])


error = mean_squared_error(y, y_hat)

print("Mean Squared Error:", error)




# Binary Cross-Entropy
def binary_cross_entropy(
    y: np.ndarray,
    y_hat: np.ndarray,
) -> float:
    # Prevent log(0), which is undefined
    epsilon = 1e-15
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

    losses = (
        y * np.log(y_hat)
        + (1 - y) * np.log(1 - y_hat)
    )

    return float(-np.mean(losses))


# Example data
y = np.array([1, 0, 1, 0])
y_hat = np.array([0.9, 0.2, 0.8, 0.1])

# Call the function
error = binary_cross_entropy(y, y_hat)

print("Binary Cross-Entropy:", error)