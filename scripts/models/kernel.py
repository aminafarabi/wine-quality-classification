import numpy as np

class RBFSVM:
    def __init__(self, gamma = None, C = 1, max_passes = 5, tol = 0.001):
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

    def compute_kernel(self, x1, x2):
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)

    def compute_kernel_matrix(self, X):
        # kernel matrix aka similarities matrix
        # we use rbf kernel that uses exponenta, the closer the K[a,b] to 1 (exp(0)) the closer the points a and b
        # gamma determines how fast the influence of a point changes with the distance
        # small gamma - more points around have influnce    
        # big gamma - only very close points have influence
        # rbf = exp(gamma*(a-b)^2)        
        sq_norms = np.sum(X**2, axis=1)
        K = (
            sq_norms[:, np.newaxis]
            + sq_norms[np.newaxis, :]
            - 2 * np.dot(X, X.T)
        )
        return np.exp(-self.gamma * K)
    '''
    def compute_kernel_matrix(self, X):
        n_rows = X.shape[0]
        kernel_matrix = np.zeros((n_rows, n_rows))
        
        for i in range(n_rows):
            for j in range(n_rows):
                
                kernel_matrix[i, j] = self.compute_kernel(X[i], X[j])
        return kernel_matrix
    '''
    def predict_row(self, i, K):
        return np.sum(self.alpha * self.y * K[:, i]) + self.b
    
    def select_j(self, i, n): # just selecting a random j that is not i
        j = i
        while j == i:
            j = np.random.randint(0, n)
        return j

    def fit(self, X, y):
        self.X = X
        self.y = y

        n_rows = X.shape[0]
        self.alpha = np.zeros(n_rows) # the "importance" of the point, > 0 - support point
        K = self.compute_kernel_matrix(X) # how similar the points are
        passes = 0 # to understrand that we're satisfied with the outcome if alphas are not changing after n of passes

        while passes < self.max_passes:
            counter = 0 # counter for passes

            for i in range(n_rows):
                Ei = self.predict_row(i, K) - self.y[i] # the error term, basically the difference between the prediction and actual class
                # tol is just a small number to avoid strict comparisons
                # error > 0 -> the prediction is too strong
                # error < 0 -> the prediction is too little
                cond1 = y[i] * Ei < -self.tol and self.alpha[i] < self.C # the model made a mistake and we need to increase the alpha
                cond2 = y[i] * Ei > self.tol and self.alpha[i] > 0 # the model is too sure and we need to decrease the alpha
                if cond1 or cond2:
                    # because we have the sum(alpha_i, y_i) = 0
                    # we need to change 2 points to save the balance
                    j = self.select_j(i, n_rows) # a random j that is not equal to i
                    Ej = self.predict_row(j, K) - self.y[j] # j'th error term

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if y[i] != y[j]: # if the classes are different the difference should be constant
                        min_alpha = max(0, self.alpha[j] - self.alpha[i]) # left threshold
                        max_alpha = min(self.C, self.C + self.alpha[j] - self.alpha[i]) # right threshold
                    else: # if the classes are the same the sum should be constant
                        min_alpha = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        max_alpha = min(self.C, self.alpha[i] + self.alpha[j])

                    if min_alpha == max_alpha:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j] # determines how much we can descend and change alpha for j

                    if eta >= 0: # mean the second derivative is not negative so no slope and we skip the update
                        continue
                    # α_j_new = α_j_old - y_j * (E_i - E_j) / η
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta # updating alpha for j, decreasing the error term
                    self.alpha[j] = np.clip(self.alpha[j], min_alpha, max_alpha) # cutting the alpha if it's out of min-max range

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j]) # after updating j, we update i

                    b1 = self.b - Ei \
                        - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]

                    b2 = self.b - Ej \
                        - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C: # if alpha i is in the margin we use it
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C: # if alpha j is in the margin we use it 
                        self.b = b2
                    else: # if both in the margin we take the average
                        self.b = (b1 + b2) / 2

                    counter += 1 # counting only when we changed alpha

            if counter == 0: 
                passes += 1 # number of passes without changing alpha, after 5 passes without change we stop
            else:
                passes = 0


    def kernel_predict(self, X):
        K = np.zeros((len(self.X), len(X)))

        for i in range(len(self.X)):
            for j in range(len(X)):
                K[i, j] = self.compute_kernel(self.X[i], X[j])

        return np.sign((self.alpha * self.y) @ K + self.b)
    
    '''
    def kernel_predict(self, X):
        y_pred = []

        for x in X:
            prediction = 0
            support_idx = self.alpha > 0
            for i in np.where(support_idx)[0]:
                if self.alpha[i] > 0:
                    prediction += self.alpha[i] * self.y[i] * self.compute_kernel(self.X[i], x)
            y_pred.append(prediction + self.b)

        return np.array(y_pred)
    '''

    def predict(self, X):
        return np.sign(self.kernel_predict(X))