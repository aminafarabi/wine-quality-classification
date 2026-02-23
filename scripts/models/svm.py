import numpy as np

# LINEAR SOFT MARGIN SVM
class SVM:
    # min 1/2 ||w||^2  + C max(0, 1 - y_i(w*x_i + b))
    # C is lambda_param (aka how many missclassifications is okay)
    def __init__(self, learning_rate, lambda_param, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        
        self.w = 0 # weights vector or normalization vector
        self.b = 0 # the shift or bias

    def fit(self, X, y):
        n_rows, n_attributes = X.shape
        
        self.w = np.zeros(n_attributes)
        self.b = 0

        # shifting the margin
        # stochastic gradient decent
        for idx in range(n_rows):
            # we take 1 row of features or 1 wine
            x_i = X[idx]
            # max(0, 1 - y_i(w*x_i + b))
            # 1 - y_i(w*x_i + b) <= 0
            # y_i(w*x_i + b) - 1 >= 0
            #  y_i(w*x_i + b) >= 1
            if y[idx] * (np.dot(self.w, x_i) + self.b) >= 1: # no error, out of margin
                # the gradient is 2*lambda*w
                # no shift, changing only w
                self.w -= self.lr * (2 * self.lambda_param * self.w)
            else: # error, in the margin
                # hinge loss y_i*x_i
                # shifting
                self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                self.b -= self.lr * (-y[idx])

    def predict(self, X):
        prediction = np.dot(X, self.w) + self.b
        return np.sign(prediction)
