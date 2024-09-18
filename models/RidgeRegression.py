import numpy as np

'''
Motivation:
    - reduce overfitting
    - increase bias to decrease variance
        => bias - inability to capture true relationship
        => variance - difference in accuracy between datasets

    - here, each theta is just slope for that feature vs. output
    - l2 regularization term is just restricting the growth of the slope
        => smaller slope means, same change in x results in less change in y
        => as alpha increases, model becomes **less sensitive to variations** of the independent variables
'''

'''
Proof for the Normal Equation for ridge regression (*):

given the cost function:
    J(Theta)    = 1/2m Sum(i=1...m) (h_theta(x^(i)) - y^(i))^2 + alpha/2m Sum(j=1...n) theta_j^2
                = 1/2m [(X ⋅ Theta - Y).T ⋅ (X ⋅ Theta - Y) + alpha ⋅ Theta.T ⋅ Theta]
        1/2m dropped as it doesn't affect minimization
                = (X ⋅ Theta - Y).T ⋅ (X ⋅ Theta - Y) + alpha ⋅ Theta.T ⋅ Theta
                = (X ⋅ Theta).T ⋅ X ⋅ Theta - (X ⋅ Theta).T ⋅ Y - Y.T ⋅ (X ⋅ Theta) + Y.T ⋅ Y + alpha ⋅ Theta.T ⋅ Theta
                = Theta.T ⋅ X.T ⋅ X ⋅ Theta - 2(X ⋅ Theta).T ⋅ Y + Y.T ⋅ Y + alpha ⋅ Theta.T ⋅ Theta

    dJ/dTheta   = 2 ⋅ X.T ⋅ X ⋅ Theta - 2 ⋅ X.T ⋅ Y + 2 ⋅ alpha ⋅ Theta = 0
                => 2 ⋅ X.T ⋅ X ⋅ Theta - 2 ⋅ X.T ⋅ Y + 2 ⋅ alpha ⋅ Theta = 0
                => X.T ⋅ X ⋅ Theta + alpha ⋅ Theta = X.T ⋅ Y
                => (X.T ⋅ X + alpha ⋅ I) ⋅ Theta = X.T ⋅ Y
                => Theta = (X.T ⋅ X + alpha ⋅ I)^-1 ⋅ X.T ⋅ Y

                => Theta = np.linalg.inv(X.T.dot(X) + alpha * np.eye(n)).dot(X.T).dot(Y)
        n = dimension of features in X
'''

'''
Proof for the gradient descent equations (**) (***):

almost the same as for linear regression, except extra term for

dJ(Theta)/dw    = 1/m * X ⋅ (y_pred - y_true) + d(1/2m * alpha * w^2)/dw
                = 1/m * X ⋅ (y_pred - y_true) + 1/m * alpha * w
                = 1/m (X ⋅ (y_pred - y_true) + alpha * w)
'''

class RidgeRegression:
    """
    Basic Implementation of the Ridge Regression Model

    L2 regularization on top of linear regression
    """

    def __init__(self, alpha=1.0, learning_rate=0.01, num_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y, grad_desc=False):
        """
        Fit the model to given data

        Args:
            X (list): list of inputs
            y (list): list of outputs
            grad_desc (bool, optional): True for gradient descent. Defaults to False.
        """

        if not grad_desc:
            X = np.insert(X, 0, 1, axis=1)

            coefficients = np.linalg.inv(X.T.dot(X) + self.alpha * np.eye(X.shape[1])).dot(X.T).dot(y)  # (*)

            self.coefficients = coefficients[1:]
            self.intercept = coefficients[0]
        else:
            num_samples, num_features = X.shape
            self.coefficients = np.zeros(num_features)
            self.intercept = 0

            for _ in range(self.num_iterations):
                y_pred = np.dot(X, self.coefficients) + self.intercept

                dw = (1 / num_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.coefficients)
                db = (1 / num_samples) * np.sum(y_pred - y)

                self.coefficients -= self.learning_rate * dw
                self.intercept -= self.learning_rate * db

    def predict(self, X):
        """
        Predicts based on given input(s)

        Args:
            X (list): list of inputs for predictions

        Returns:
            list: list of outputs
        """

        return np.dot(X, self.coefficients) + self.intercept
    
    def score(self, X, y):
        """
        R-squared of the model

        Args:
            X (list): list of inputs
            y (list): true outputs
        """

        y_pred = self.predict(X)
        diff = y - y_pred

        sum_diff_squared = np.sum(diff ** 2)
        sum_squared = np.sum((y - np.mean(y)) ** 2)

        return 1 - (sum_diff_squared / sum_squared)