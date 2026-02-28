import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, lambda_param, n_iters=1000, class_weight=None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500) # to avoid overflow
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_rows, n_features = X.shape
        
        # converting y from -1/1 to 0/1 
        # binary cross-entropy loss uses 0/1
        y_binary = np.where(y == -1, 0, 1)

        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        # balancing by implementing weight cause the data is inbalanced
        n_pos = np.sum(y_binary == 1)
        n_neg = np.sum(y_binary == 0)
        weight_pos = n_rows / (2 * n_pos)
        weight_neg = n_rows / (2 * n_neg)
        sample_weights = np.where(y_binary == 1, weight_pos, weight_neg)
        
        
        for _ in range(self.n_iters):
            z = np.dot(X, self.weights) + self.bias #logits
            y_pred = self.sigmoid(z) # getting probability
            
            # L = -1/N * sum([y_pred*log(y) + (1-y_pred)*log(1-y)])
            loss = (-1/n_rows) * np.sum(
                sample_weights * (
                    y_binary * np.log(y_pred + 1e-8) + 
                    (1 - y_binary) * np.log(1 - y_pred + 1e-8) # to avoid deleting by 0
                )
            )
            # (lambda/2) * sum(w**2)
            reg_loss = (self.lambda_param/2) * np.sum(self.weights**2)

            # gradients
            # (1/N) * X *(y_pred - y) + lambda*w
            dw = (1/n_rows) * np.dot(X.T, (y_pred - y_binary)) + self.lambda_param * self.weights
            # (1/N) * sum(y_pred - y)
            db = (1/n_rows) * np.sum(y_pred - y_binary)
            
            dw_norm = np.linalg.norm(dw)
            if dw_norm > 1: # if gradient is >1 we normalise to 1 
                dw = dw / dw_norm
            
            # updating w and b
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return np.where(probabilities >= threshold, 1, -1)