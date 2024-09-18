import numpy as np

'''
Proof for the Normal Equation for linear regression (*):

given the hypothesis function:
    h_theta(x)  = theta_0 * x_0 + theta_1 * x_1 + ... + theta_n * x_n
                = Theta.T ⋅ x

given the least-squares cost function:
    J(Theta)    = 1/2m Sum(i=1...m) (h_theta(x^(i)) - y^(i))^2
        (h_theta(x^(i)) - y^(i))^2 - difference between prediction and actual value squared
        Sum(i=1...m) - sum of all differences squared
        1/2m - average * 1/2 (constant multiplied as convention to get rid of the 2 after derivation?)
                = 1/2m (X ⋅ Theta - Y).T ⋅ (X ⋅ Theta - Y)
        X - m by n matrix, m = # of samples, n = input dimension

dropping 1/2m (comparing to 0 in the end so constants don't matter):
    J(Theta)    = ((X ⋅ Theta).T - Y.T) ⋅ (X ⋅ Theta - Y)
                = (X ⋅ Theta).T ⋅ X ⋅ Theta - (X ⋅ Theta).T ⋅ Y - Y.T ⋅ (X ⋅ Theta) + Y.T ⋅ Y
        combine term 2 & 3, order can be changed because both are vectors
                = Theta.T ⋅ X.T ⋅ X ⋅ Theta - 2(X ⋅ Theta).T ⋅ Y + Y.T ⋅ Y

take partial derivative with respect to Theta (want to minimize J(Theta), so look for derivative = 0):
    dJ/dTheta   = 2 X.T ⋅ X ⋅ Theta - 2 X.T ⋅ Y = 0
                => X.T ⋅ X ⋅ Theta = X.T ⋅ Y
                => Theta = (X.T ⋅ X)^-1 ⋅ X.T . Y

                => Theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
'''

'''
Proof for the gradient descent equations (**) (***):

given the least-squares cost function:
    J(Theta)    = 1/2m Sum(i=1...m) (h_theta(x^(i)) - y^(i))^2

dJ(Theta)/dw    = 1/2m * 2 * [Sum(i=1...m) (y_pred - y_true)] * d(y_pred)/dw
                = 1/m * [Sum(i=1...m) (y_pred - y_true)] * d(w ⋅ X + b)/dw
                = 1/m * [Sum(i=1...m) (y_pred - y_true)] * X
                = 1/m * X ⋅ (y_pred - y_true)

similarly,
dJ(Theta)/db    = 1/m * [Sum(i=1...m) (y_pred - y_true)] * d(w ⋅ X + b)/db
                = 1/m * [Sum(i=1...m) (y_pred - y_true)]
'''

class LinearRegression:
    """Basic Implementation of the Linear Regression Model""" 

    def __init__(self, learning_rate=0.01, num_iterations=1000):
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
        """        

        if not grad_desc:
            '''
            n input variables => dim(X) should be n+1, add intercept as first value
              X => list to insert to
              0 => insert before index 0
              1 => insert value 1
              axis=1  => insert for every row, basically insert a column
            '''
            X = np.insert(X, 0, 1, axis=1)

            coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) # (*)

            self.coefficients = coefficients[1:]
            self.intercept = coefficients[0]
        else:
            # initialize some variables
            num_samples, num_features = X.shape
            self.coefficients = np.zeros(num_features)
            self.intercept = 0

            for _ in range(self.num_iterations):
                # perform gradient descent
                y_pred = np.dot(X, self.coefficients) + self.intercept

                dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / num_samples) * np.sum(y_pred - y)

                self.coefficients -= self.learning_rate * dw    # (**)
                self.intercept -= self.learning_rate * db       # (***)

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