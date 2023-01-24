import numpy as np


class LinearRegression:
    """
    A linear regression model based on closed-form solution
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = None
        self.b = None
        self.params = None
        self.has_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """fit linear regression model on the data based on closed-form solution

        Args:
            X (np.ndarray): the input data
            y (np.ndarray): the label of the data
        """
        self.has_fitted = True

        ## get the bias term
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        ## closed-form solution
        inv_xTx = np.linalg.inv(np.dot(X.T, X))
        self.params = np.dot(inv_xTx, np.dot(X.T, y))

        ## store results
        self.b = self.params[-1]
        self.w = self.params[:-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make prediction based on the input data and the fitted data

        Args:
            X (np.ndarray): the input data

        Returns:
            pred (np.ndarray): the prediction based on the input data and the fitted regression model.
        """
        if self.has_fitted == False:
            raise Exception("The linear regression model hasn't been fitted yet.")

        pred = self.b + np.dot(X, self.w)
        return pred


class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """fit linear regression model on the data with gradient descent

        Args:
            X (np.ndarray): the input data
            y (np.ndarray): the label of the data
            lr (float, optional): learning rate for gradient descent. Defaults to 0.01.
            epochs (int, optional): number of iterations. Defaults to 1000.
        """

        self.has_fitted = True

        ## initiatize weights and bias for graidient descent
        self.w = np.random.rand(len(y))
        self.b = 0

        for i in range(epochs):
            w_grad = 0.0
            b_grad = 0.0

            ## update gradient based on gradient descent
            for n in range(len(X)):
                w_grad = w_grad - 2.0 * (y[n] - self.b - self.w * X[n]) * X[n]
                b_grad = b_grad - 2.0 * (y[n] - self.b - self.w * X[n]) * 1.0

            ## update parameters based on gradient and learning rate
            self.w = self.w - lr * w_grad
            self.b = self.b - lr * b_grad

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): the input data.

        Returns:
            pred (np.ndarray): the prediction based on the input data and the fitted regression model.

        """
        if self.has_fitted == False:
            raise Exception("The linear regression model hasn't been fitted yet.")

        pred = self.b + np.dot(X, self.w)
        return pred
