import numpy as np

class Kernel_SVM:
    # min 1/2 ||w||^2  + C max(0, 1 - y_i(w*x_i + b))
    # C is lambda_param (aka how many missclassifications is okay)
    def __init__(self, kernel="rbf", C=1.0, gamma=0.1, tol=1e-3, max_passes=5):
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes

    def kernel(self, x1, x2): # kernel matrix aka similarities matrix
        if self.kernel_name == "linear":
            return np.dot(x1, x2)
        elif self.kernel_name == "rbf":
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel")

    def compute_kernel_matrix(self, X):
        n_rows = X.shape[0]
        K = np.zeros((n_rows, n_rows))
        
        for i in range(n_rows):
            for j in range(n_rows):
                K[i, j] = self.kernel(X[i], X[j])
        
        return K

    def fit(self, X, y):
        n_rows, n_features = X.shape
        
        self.X = X
        self.y = y
        self.alpha = np.zeros(n_rows) # the "importance" of the point, >0 - suppurt point
        self.b = 0

        K = self.compute_kernel_matrix(X)

        passes = 0

        while passes < self.max_passes:
            num_changed = 0

            for i in range(n_rows):
                Ei = self._predict_row(i, K) - y[i]

                if (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * Ei > self.tol and self.alpha[i] > 0):

                    j = self._select_j(i, n_rows)
                    Ej = self._predict_row(j, K) - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.b - Ei \
                        - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]

                    b2 = self.b - Ej \
                        - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] \
                        - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

    def _predict_row(self, i, K):
        return np.sum(self.alpha * self.y * K[:, i]) + self.b

    def _select_j(self, i, n_rows):
        j = i
        while j == i:
            j = np.random.randint(0, n_rows)
        return j

    def project(self, X):
        y_pred = []

        for x in X:
            s = 0
            for i in range(len(self.alpha)):
                if self.alpha[i] > 0:
                    s += self.alpha[i] * self.y[i] * self.kernel(self.X[i], x)
            y_pred.append(s + self.b)

        return np.array(y_pred)

    def predict(self, X):
        return np.sign(self.project(X))