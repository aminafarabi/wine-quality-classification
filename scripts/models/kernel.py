import numpy as np

class RBFSVM:
    def __init__(self, gamma = None, C = 1, max_passes = 2, tol = 0.001):
        # RBF SVM parameters
        self.max_passes = max_passes
        self.gamma = gamma # adding rbf parameter
        self.C = C # the alpha max value [0,C]
        self.tol = tol # a small number for avoiding very strict comparisons with 0

        # common parameters
        self.b = 0 # the shift or bias
        self.alpha = None
        self.X = None
        self.y = None

        self.K = None
        self.errors = None
        #self.cache_size = 500  # maximum number of kernel rows to cache
        #self.kernel_cache = {}  # cache for kernel computations

    def compute_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def compute_kernel_matrix(self, X):
        # kernel matrix aka similarities matrix
        # we use rbf kernel that uses exponenta, the closer the K[a,b] to 1 (exp(0)) the closer the points a and b
        # gamma determines how fast the influence of a point changes with the distance
        # small gamma - more points around have influnce    
        # big gamma - only very close points have influence
        # rbf = exp(gamma*(a-b)^2)        
        squared_norms = np.sum(X**2, axis=1, keepdims=True)
        K = squared_norms + squared_norms.T - 2 * np.dot(X, X.T)
        K = np.maximum(K, 0)
        return np.exp(-self.gamma * K)
    
    def predict_row(self, i):
        return np.sum(self.alpha * self.y * self.K[:, i]) + self.b
    
    def select_j(self, i, n): # just selecting a random j that is not i
        j = i
        while j == i:
            j = np.random.randint(0, n)
        return j

    def predict_batch(self, X, alpha, y, b, K_row=None):
        if K_row is not None:
            return np.dot(alpha * y, K_row) + b
        return np.dot(alpha * y, X) + b
    
    def select_j_heuristic(self, i, Ei, n_rows, alpha, errors, C):
        mask = np.ones(n_rows, dtype=bool)
        mask[i] = False
        
        if Ei > 0:
            candidate_mask = mask & (errors < 0) & (alpha < C)
        else:
            candidate_mask = mask & (errors > 0) & (alpha > 0)
        
        candidates = np.where(candidate_mask)[0]
        
        if len(candidates) > 0:
            # candidates with maximum |Ei - Ej|
            diffs = np.abs(Ei - errors[candidates])
            return candidates[np.argmax(diffs)]
        # random if no candidates
        j = i
        while j == i:
            j = np.random.randint(0, n_rows)
        return j

    def update_errors_batch(self, indices, alpha, y, K, b):
        predictions = np.dot(alpha * y, K[:, indices]) + b
        return predictions - y[indices]

    def update_error(self, i):
        self.errors[i] = np.sum(self.alpha * self.y * self.K[:, i]) + self.b - self.y[i]
        return self.errors[i]

    def fit(self, X, y):
        self.X = X
        self.y = y

        # balancing the errors to punish -1 missclassification more
        n_samples = len(y)
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)
        weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1
        weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1
        self.sample_weights = np.where(y == 1, weight_pos, weight_neg)
        self.C_weighted = self.C * self.sample_weights

        n_rows = X.shape[0]
        self.alpha = np.zeros(n_rows) # the "importance" of the point, > 0 - support point
       
        self.K = self.compute_kernel_matrix(X) # how similar the points are
        self.errors = np.zeros(n_rows)
        for i in range(n_rows):
            self.errors[i] = self.predict_row(i) - y[i]

        passes = 0 # to understrand that we're satisfied with the outcome if alphas are not changing after n of passes

        while passes < self.max_passes:
            counter = 0 # counter for passes

            for i in range(n_rows):
                Ei = self.predict_row(i) - self.y[i] # the error term, basically the difference between the prediction and actual class
                # tol is just a small number to avoid strict comparisons
                # error > 0 -> the prediction is too strong
                # error < 0 -> the prediction is too little
                #cond1 = y[i] * Ei < -self.tol and self.alpha[i] < self.C # the model made a mistake and we need to increase the alpha
                #cond2 = y[i] * Ei > self.tol and self.alpha[i] > 0 # the model is too sure and we need to decrease the alpha

                cond1 = self.y[i] * Ei < -self.tol and self.alpha[i] < self.C_weighted[i]
                cond2 = self.y[i] * Ei > self.tol and self.alpha[i] > 0
                if cond1 or cond2:
                    # because we have the sum(alpha_i, y_i) = 0
                    # we need to change 2 points to save the balance
                    j = self.select_j_heuristic(i, Ei, n_rows, self.alpha, self.errors, self.C_weighted)
                    #Ej = self.predict_row(j) - self.y[j] # j'th error term
                    Ej = self.errors[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if y[i] != y[j]: # if the classes are different the difference should be constant
                        min_alpha = max(0, self.alpha[j] - self.alpha[i]) # left threshold
                        max_alpha = min(self.C_weighted[j], self.C_weighted[j] + self.alpha[j] - self.alpha[i]) # right threshold
                    else: # if the classes are the same the sum should be constant
                        min_alpha = max(0, self.alpha[i] + self.alpha[j] - self.C_weighted[j])
                        max_alpha = min(self.C_weighted[j], self.alpha[i] + self.alpha[j])

                    if min_alpha == max_alpha:
                        continue

                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j] # determines how much we can descend and change alpha for j

                    if eta >= 0: # mean the second derivative is not negative so no slope and we skip the update
                        continue
                    # α_j_new = α_j_old - y_j * (E_i - E_j) / η
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta # updating alpha for j, decreasing the error term
                    self.alpha[j] = np.clip(self.alpha[j], min_alpha, max_alpha) # cutting the alpha if it's out of min-max range

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j]) # after updating j, we update i

                    b1 = self.b - Ei \
                        - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]

                    b2 = self.b - Ej \
                        - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

                    if 0 < self.alpha[i] < self.C_weighted[i]: # if alpha i is in the margin we use it
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C_weighted[j]: # if alpha j is in the margin we use it 
                        self.b = b2
                    else: # if both in the margin we take the average
                        self.b = (b1 + b2) / 2

                    #self.errors[i] = np.sum(self.alpha * self.y * self.K[:, i]) + self.b - self.y[i]
                    #self.errors[j] = np.sum(self.alpha * self.y * self.K[:, j]) + self.b - self.y[j]

                    #self.errors[i] = self.update_error(i)
                    #self.errors[j] = self.update_error(j)

                    self.errors[[i, j]] = self.update_errors_batch(
                        [i, j], self.alpha, self.y, self.K, self.b
                    )

                    counter += 1 # counting only when we changed alpha

            #print(f"    Pass {passes + 1}/{self.max_passes}, updates: {counter}")
            if counter == 0: 
                passes += 1 # number of passes without changing alpha, after 5 passes without change we stop
            else:
                passes = 0


    def predict(self, X):
        # support vectors
        sv_mask = self.alpha > 1e-5
        
        if not np.any(sv_mask):
            return np.ones(len(X)) * np.sign(self.b)
        
        sv_alpha = self.alpha[sv_mask]
        sv_y = self.y[sv_mask]
        sv_X = self.X[sv_mask]
        
        diff = sv_X[:, np.newaxis, :] - X[np.newaxis, :, :]
        squared_distances = np.sum(diff ** 2, axis=2)
        K_pred = np.exp(-self.gamma * squared_distances)
        
        predictions = np.dot(sv_alpha * sv_y, K_pred) + self.b
        return np.sign(predictions)
