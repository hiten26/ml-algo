import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Gradient Descent algorithm to find the optimal weights and bias
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Make predictions using the current weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate the gradients of weights and bias with respect to the loss function
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            # Update weights and bias using gradient descent
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Make predictions using the trained model
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":

    # Sample data for demonstration
    X = np.array([[1], [2], [3], [4], [5]])  # Input features (e.g., the x values)
    y = np.array([2, 4, 5, 4, 5])            # Target values (e.g., the corresponding y values)

    # Create and train the linear regression model
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    # Make predictions
    test_X = np.array([[6], [7]])
    predictions = model.predict(test_X)

    print("Predictions:", predictions)