import numpy as np

# accuracy
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# precision, the number of true-positive over all-positive
def precision(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)

# recall, the number of misclassified positives
def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn + 1e-8)

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)


def stratified_k_fold_split(y, k=5):
    classes = [-1, 1]

    class_indeces = {}

    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        class_indeces[c] = idx
    
    folds = [[] for _ in range(k)] # because we need to use extend function

    for c in classes:
        idx_by_class = class_indeces[c]
        # deviding by k
        idx_by_class_splits = np.array_split(idx_by_class, k)
        for i in range(k):
            folds[i].extend(idx_by_class_splits[i])

    for i in range(k):
        folds[i] = np.array(folds[i]) # so we can shuffle and hstack in cross-validation
        np.random.shuffle(folds[i])    

    return folds # matrix of random 5 groups of indeces

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
        
    n_rows = X.shape[0]
    indices = np.random.permutation(n_rows)
    
    test_count = int(n_rows * test_size)
    
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    
    return X_train, X_test, y_train, y_test

def cross_validate(model_class, X, y, k=5, **model_params):
    folds = stratified_k_fold_split(y, k)
    scores = {"Accuracy": [],
              "Precision": [],
              "Recall": [],
              "F1_score": [],
              }

    for i in range(k):
        test_idx = folds[i]
        # taking all folds except i-th
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores["Accuracy"].append(accuracy(y_test, y_pred))
        scores["Precision"].append(precision(y_test, y_pred))
        scores["Recall"].append(recall(y_test, y_pred))
        scores["F1_score"].append(f1_score(y_test, y_pred))
    
    
    scores["Accuracy"] = np.mean(scores["Accuracy"])
    scores["Precision"] = np.mean(scores["Precision"])
    scores["Recall"] = np.mean(scores["Recall"])
    scores["F1_score"] = np.mean(scores["F1_score"])
    
    return scores


param_ranges = {
    "lr": (0.0001, 0.1),
    "lambda_param": (0.0001, 0.1),
    "n_iters": (100, 1000)
}

def random_search(model_class, X, y, k=5, n_trials = 50):
    best_score = -1
    best_params = None

    for n in range(n_trials):
        #choosing randomly
        lr = 10 ** np.random.uniform(np.log10(param_ranges["lr"][0]), np.log10(param_ranges["lr"][1]))
        lambda_param = 10 ** np.random.uniform(np.log10(param_ranges["lambda_param"][0]), np.log10(param_ranges["lambda_param"][1]))
        n_iters = np.random.randint(param_ranges["n_iters"][0], param_ranges["n_iters"][1]+1)

        # folds = stratified_k_fold_split(X, y, k=k)
        
        scores = cross_validate(
                        model_class,
                        X,
                        y,
                        k=k,
                        learning_rate=lr,
                        lambda_param=lambda_param,
                        n_iters=n_iters
                    )
        f1_score = scores["F1_score"]
        accuracy = scores["Accuracy"]
        #print(f"lr={lr}, lambda={lambda_param}, n_iters={n_iters} → {f1_score}")

        if f1_score > best_score:
            best_score = f1_score
            best_params = {
                "learning_rate": lr,
                "lambda_param": lambda_param,
                "n_iters": n_iters
                }
        '''
        for lr in param_grid["learning_rate"]:
            for lambda_param in param_grid["lambda_param"]:
                for n_iters in param_grid["n_iters"]:

                    scores = cross_validate(
                        model_class,
                        X,
                        y,
                        k=k,
                        learning_rate=lr,
                        lambda_param=lambda_param,
                        n_iters=n_iters
                    )
                    print("------------------------------------------------")
                    print(f"lr={lr}, lambda={lambda_param}, n_iters={n_iters} →")
                    for metric, value in scores.items():
                        print(f"{metric}: {value:.4f}")
                    
                    if scores["Accuracy"] > best_score:
                        best_score = scores["Accuracy"]
                        best_params = {
                            "learning_rate": lr,
                            "lambda_param": lambda_param,
                            "n_iters": n_iters
                        }
        '''

    return best_params, best_score


