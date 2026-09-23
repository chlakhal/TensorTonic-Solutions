import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """


def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """

    # Convert inputs to NumPy arrays
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Number of training samples
    n = len(y)

    # One weight per feature/column
    W = np.zeros(X.shape[1], dtype=np.float64)

    # Bias
    b = 0.0

    # Batch gradient descent
    for _ in range(epochs):

        # Predictions
        y_pred = X @ W + b

        # Prediction error
        error = y_pred - y

        # Gradients
        dW = (2 / n) * (X.T @ error)
        db = (2 / n) * np.sum(error)

        # Gradient descent update
        W -= lr * dW
        b -= lr * db

    # Return rounded results
    return np.round(W, 4).tolist(), round(float(b), 4)