import numpy as np


class LinearRegression:
    """
    A linear regression model that uses minimum OLS regression loss.
    """

    def __init__(self):
        """
        Initializing the weight w and the intercept b.

        Arguments:
            w (np.ndarray): Weight.
            b (float): Intercept.

        Returns:
            None
        """
        self._theta = None
        self.w = None
        self.b = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Using the formula to calculate w and b.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input data.

        Returns:
            None
        """
        X = np.hstack([np.ones((len(X), 1)), X])

        self._theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

        self.b = self._theta[0]
        self.w = self._theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.hstack([np.ones((len(X), 1)), X])
        return X.dot(self._theta)


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Using for loop to get the gradient under given learning rate and epochs.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The input data.
            lr (float): The learning rate
            epochs (int): Number of iterations

        Returns:
            None
        """
        X = np.hstack([np.ones((len(X), 1)), X])
        theta = np.zeros(X.shape[1])

        for i in range(epochs):
            gradient = X.T.dot(X.dot(theta) - y) * 2.0 / len(y)
            theta -= lr * gradient

        self._theta = theta

        self.b_ = self._theta[0]
        self.w_ = self._theta[1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        X = np.hstack([np.ones((len(X), 1)), X])
        return X.dot(self._theta)
