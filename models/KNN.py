import numpy as np

class KNN:
    """
    Basic Implementation of K-Nearest Neighbors Algorithm

    Predicting based off of k nearest neighbors determined by Euclidean distance
    """

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        """
        KNN has no parameters, so fit just stores target X and y

        Args:
            X (list): list of inputs
            y (list): list of outputs
        """

        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predicts based on given input(s)

        Args:
            X (list): list of inputs for predictions

        Returns:
            list: list of outputs
        """

        return np.array([self._predict(x) for x in X])

    def _predict(self, X):
        """
        Outputs coefficients for next round of coordinate descent

        Args:
            x (list): input features 

        Returns:
            float: predictions
        """

        distances = np.sum(np.power(self.X_train - X, 2), axis=1)
        sorted = np.argsort(distances)[:self.k]
        prediction = np.average(self.y_train[sorted])

        return prediction

