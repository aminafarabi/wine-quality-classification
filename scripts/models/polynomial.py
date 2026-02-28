import numpy as np

class PolynomialSVM:
    def __init__(self, degree=2, gamma=None, coef0=1, C=1, max_passes=2, tol=0.001):
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.C = C
        self.max_passes = max_passes
        self.tol = tol
        
        self.b = 0
        self.alpha = None
        self.X = None
        self.y = None
        self.K = None
        self.errors = None
        self.sample_weights = None
        self.C_weighted = None
        
    def compute_kernel_matrix(self, X):
        if self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        
        # Compute dot product
        dot_matrix = np.dot(X, X.T)
        
        # Scale dot product to [-1, 1] range
        # Find max absolute value
        max_dot = np.abs(dot_matrix).max()
        if max_dot > 0:
            dot_matrix = dot_matrix / max_dot
        
        # Now dot_matrix is in [-1, 1]
        K = (self.gamma * dot_matrix + self.coef0) ** self.degree
        
        # Additional normalization to [0, 1] range
        K_min = K.min()
        K_max = K.max()
        if K_max > K_min:
            K = (K - K_min) / (K_max - K_min)
        
        # Clip for safety
        K = np.clip(K, 1e-10, 1e10)
        
        return K
    
    def predict_row(self, i):
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b
    
    def select_j_heuristic(self, i, Ei, n_rows, alpha, errors, C_weighted):
        mask = np.ones(n_rows, dtype=bool)
        mask[i] = False
        
        if Ei > 0:
            candidate_mask = mask & (errors < 0) & (alpha < C_weighted)
        else:
            candidate_mask = mask & (errors > 0) & (alpha > 0)
        
        candidates = np.where(candidate_mask)[0]
        
        if len(candidates) > 0:
            diffs = np.abs(Ei - errors[candidates])
            return candidates[np.argmax(diffs)]
        
        j = i
        while j == i:
            j = np.random.randint(0, n_rows)
        return j
    
    def update_errors_batch(self, indices, alpha, y, K, b):
        predictions = np.dot(alpha * y, K[:, indices]) + b
        return predictions - y[indices]
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        n_rows = X.shape[0]
        
        n_samples = len(y)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)
        
        weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1
        weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1
        
        self.sample_weights = np.where(y == 1, weight_pos, weight_neg)
        self.C_weighted = self.C * self.sample_weights
        
        self.alpha = np.zeros(n_rows)
        self.b = 0
        
        self.K = self.compute_kernel_matrix(X)
        
        # Initialize errors
        self.errors = np.zeros(n_rows)
        for i in range(n_rows):
            self.errors[i] = self.predict_row(i) - y[i]
        
        passes = 0
        
        while passes < self.max_passes:
            counter = 0
            
            for i in range(n_rows):
                Ei = self.errors[i]
                
                cond1 = self.y[i] * Ei < -self.tol and self.alpha[i] < self.C_weighted[i]
                cond2 = self.y[i] * Ei > self.tol and self.alpha[i] > 0
                
                if cond1 or cond2:
                    j = self.select_j_heuristic(
                        i, Ei, n_rows, self.alpha, self.errors, self.C_weighted
                    )
                    Ej = self.errors[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C_weighted[j], self.C_weighted[j] + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C_weighted[j])
                        H = min(self.C_weighted[j], self.alpha[i] + self.alpha[j])
                    
                    if L >= H:
                        continue
                    
                    # Compute eta with safety check
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    
                    if abs(eta) < 1e-12:  # Too small, skip
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - Ei \
                         - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    
                    b2 = self.b - Ej \
                         - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                    
                    if 0 < self.alpha[i] < self.C_weighted[i]:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C_weighted[j]:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    # Update errors
                    self.errors[[i, j]] = self.update_errors_batch(
                        [i, j], self.alpha, self.y, self.K, self.b
                    )
                    
                    counter += 1
            
            if counter == 0:
                passes += 1
            else:
                passes = 0
            
            if passes % 1 == 0 and counter > 0:
                sv_count = np.sum(self.alpha > 1e-5)
    
    def predict(self, X): 
        sv_mask = self.alpha > 1e-5
        
        if not np.any(sv_mask):
            return np.ones(len(X)) * np.sign(self.b)
        
        sv_alpha = self.alpha[sv_mask]
        sv_y = self.y[sv_mask]
        sv_X = self.X[sv_mask]
        
        # Normalize dot products the same way as training
        dot_product = np.dot(sv_X, X.T)
        
        # Apply same scaling as training
        max_dot = np.abs(np.dot(self.X, self.X.T)).max()
        if max_dot > 0:
            dot_product = dot_product / max_dot
        
        K_pred = (self.gamma * dot_product + self.coef0) ** self.degree
        
        # Normalize to match training kernel range
        # This is approximate but helps
        K_pred = np.clip(K_pred, 1e-10, 1e10)
        
        predictions = np.dot(sv_alpha * sv_y, K_pred) + self.b
        return np.sign(predictions)