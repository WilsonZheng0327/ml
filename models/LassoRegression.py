import numpy as np

'''
Motivation:
    - feature selection
        => shrinking coefficients to zero, identify less important predictors
    - why is feature selection important?
        => more features means more flexibility for the model
        => more flexiblility usually means better fit to training data
           but maybe not all features are important predictors for the output
           so these features are just providing new degrees of freedom
           not necessarily helping to generalize, but overfitting to training data
        => hence feature selection is important
'''

class LassoRegression:
    """
    Basic Implementation of the LASSO (Least Absolute Shrinkage and Selection Operator) Regression Model

    L1 regularization on top of linear regression
    """

    def __init__(self, alpha=1.0, max_iter=1000, tolerance=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the model to given data

        Args:
            X (list): list of inputs
            y (list): list of outputs
        """

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        # cyclical coordinate descent
        for _ in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                X_j = X[:, j]   # j-th feature column
                y_pred = self.predict(X)

                # coordinate descent equation
                r_j = y - y_pred + self.coef_[j] * X_j
                z_j = np.dot(X_j, r_j) / n_samples

                # update coefficients
                self.coef_[j] = self._thresholding(z_j, self.alpha / (2 * n_samples))

            # update intercept
            self.intercept_ = np.mean(y - np.dot(X, self.coef_))

            # check changes of coefficients
            if np.sum(np.abs(self.coef_ - coef_old)) < self.tolerance:
                break

    def predict(self, X):
        """
        Predicts based on given input(s)

        Args:
            X (list): list of inputs for predictions

        Returns:
            list: list of outputs
        """

        return np.dot(X, self.coef_) + self.intercept_

    def _thresholding(self, x, lambda_):
        """
        Outputs coefficients for next round of coordinate descent

        Args:
            x (double): input 
            lambda_ (int): threshold

        Returns:
            double: new coefficient
        """

        if x > lambda_:
            return x - lambda_
        elif x < -lambda_:
            return x + lambda_
        else:
            return 0