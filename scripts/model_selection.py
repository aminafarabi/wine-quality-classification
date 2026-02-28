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

# f1 score works better for inbalanced classes
def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)

#------------------------------------------------------------------------------------------------
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

def stratified_k_fold_split_with_stats(X, y, k=5):
    classes = [-1, 1]
    n_features = X.shape[1]

    class_indices = {}

    for c in classes:
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        class_indices[c] = idx
    
    folds = []
    for _ in range(k):
        fold = {
            "indices": [],
            "X": [],
            "mean": np.zeros(n_features),
            "var": np.zeros(n_features),
            "n": 0,
        }
        folds.append(fold)

    for c in classes:
        idx_by_class = class_indices[c]
        idx_by_class_splits = np.array_split(idx_by_class, k)
        for i in range(k):
            folds[i]["indices"].extend(idx_by_class_splits[i])

    for i in range(k):
        folds[i]["indices"] = np.array(folds[i]["indices"])
        np.random.shuffle(folds[i]["indices"])
        
        folds[i]["X"] = X[folds[i]["indices"]]
        folds[i]["y"] = y[folds[i]["indices"]]

        folds[i]["mean"] = np.mean(folds[i]["X"], axis=0)
        folds[i]["var"] = np.var(folds[i]["X"], axis=0)
        folds[i]["n"] = len(folds[i]["indices"])
    
    return folds

def prepare_cv_folds(X, y, k=5):
    # pre-compute alll possible folds in cv to save time
   
    base_folds = stratified_k_fold_split_with_stats(X, y, k) # list of 5 folds
    cv_folds = []
    for test_idx in range(k):
        test_data = { # 1 fold is test, others are combined
            'X': base_folds[test_idx]['X'],
            'y': base_folds[test_idx]['y']
        }
        
        # combining
        total_n = 0
        weighted_mean_sum = 0
        weighted_var_sum = 0
        train_data_X = []
        train_data_y = []
        
        for j in range(k):
            if j != test_idx:
                train_data_X.append(base_folds[j]['X'])
                train_data_y.append(base_folds[j]['y'])
                
                # computing statistics
                n_j = base_folds[j]['n']
                mean_j = base_folds[j]['mean']
                var_j = base_folds[j]['var']
                
                total_n += n_j
                weighted_mean_sum += n_j * mean_j
                weighted_var_sum += n_j * (var_j + mean_j**2)
        
        # combined statistics
        train_mean = weighted_mean_sum / total_n
        train_var = (weighted_var_sum / total_n) - train_mean**2
        train_std = np.sqrt(np.abs(train_var))
        train_std = np.where(train_std < 1e-8, 1, train_std)
        
        X_train_raw = np.vstack(train_data_X)
        y_train_raw = np.hstack(train_data_y)

        X_train_scaled = (X_train_raw - train_mean) / train_std
        
        # scaling test data using training statistics!!!!!
        X_test_scaled = (test_data['X'] - train_mean) / train_std
        
        # Create fold structure
        fold = {
            'train': {
                'X': X_train_scaled,
                'y': y_train_raw,
                'mean': train_mean,  # optional
                'std': train_std     # optional
            },
            'test': {
                'X': X_test_scaled,
                'y': test_data['y'],
            }
        }
        
        cv_folds.append(fold)
    
    return cv_folds


def cross_validate(model_class, folds, **model_params):
    f1_list = []

    for i, fold in enumerate(folds):
        #print("CV:", i)
        X_train = fold['train']['X']
        y_train = fold['train']['y']
        X_test = fold['test']['X']
        y_test = fold['test']['y']
        
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        f1_list.append(f1)
    
    f1 = np.mean(f1_list)
       
    return f1

param_ranges = {
    "lr": (0.0001, 0.1),
    "lambda_param": (0.0001, 0.01),
    "n_iters": (200, 1500),
    "gamma": (0.01, 1),
    "C": (1, 100)
}

def random_search(model_class, X, y, k=5, n_trials=30):
    # Random search for hyperparameter tuning
    # model_class: Either LinearSVM or RBFSVM class (not an instance)
    # X: training data
    # y: labels
    # k: number of folds for cross-validation
    # n_trials: number of random parameter combinations to try

    best_score = -1
    best_params = None
    folds = prepare_cv_folds(X, y, k)
    for n in range(n_trials):
        # checking model class we're using
        if model_class.__name__ == "LinearSVM" or model_class.__name__ == "LogisticRegression":

            lr = 10 ** np.random.uniform(np.log10(param_ranges["lr"][0]), np.log10(param_ranges["lr"][1]))
            lambda_param = 10 ** np.random.uniform(np.log10(param_ranges["lambda_param"][0]), np.log10(param_ranges["lambda_param"][1]))
            n_iters = np.random.randint(param_ranges["n_iters"][0], param_ranges["n_iters"][1]+1)

            params = {
                "learning_rate": lr,
                "lambda_param": lambda_param,
                "n_iters": n_iters
            }

            #print(f"Trial {n+1}/{n_trials}: lr={lr:.6f}, lambda={lambda_param:.6f}, n_iters={n_iters}")

        elif model_class.__name__ == "RBFSVM" or model_class.__name__ == "PolynomialSVM":
            gamma = 10 ** np.random.uniform(np.log10(param_ranges["gamma"][0]), np.log10(param_ranges["gamma"][1]))
            c_value = 10 ** np.random.uniform(np.log10(param_ranges["C"][0]), np.log10(param_ranges["C"][1]))

            params = {
                "gamma": gamma,
                "C": c_value
            }
            print(f"Trial {n+1}/{n_trials}: gamma={gamma:.6f}, C={c_value:.6f}")

        else:
            raise ValueError(f"Unknown model class: {model_class.__name__}. Please use LinearSVM or RBFSVM")
        
        # CV
        f1_score = cross_validate(model_class, folds, **params)
        print(f"  → F1={f1_score:.4f}")

        if f1_score > best_score:
            best_score = f1_score
            best_params = params
        
    return best_params, best_score

'''
def coarse_to_fine_tuning(model_class, X, y, k=5, total_trials=30):
    """
    Simple 3-phase tuning strategy:
    Phase 1: Wide search - explore big range
    Phase 2: Medium search - narrow around best
    Phase 3: Fine search - fine-tune around best
    
    Returns: (best_params, best_score)
    """
    
    # Store original ranges to restore later
    original_ranges = param_ranges.copy()
    
    best_overall_score = -1
    best_overall_params = None
    
    # PHASE 1: Wide exploration (40% of trials)
    print("\n" + "="*50)
    print("PHASE 1: Wide exploration (log scale)")
    print("="*50)
    
    # For RBF SVM, use log scale for gamma and C (multiply/divide by 100)
    if model_class.__name__ == "RBFSVM":
        param_ranges["gamma"] = (0.001, 100)  # Wide range: 0.001 to 100
        param_ranges["C"] = (0.1, 1000)       # Wide range: 0.1 to 1000
        print(f"Trying gamma from 0.001 to 100")
        print(f"Trying C from 0.1 to 1000")
    
    # Run phase 1
    n_phase1 = total_trials // 3  # ~10 trials
    print(f"\nTrying {n_phase1} random combinations...")
    params1, score1 = random_search(model_class, X, y, k, n_phase1)
    print(f"Phase 1 best: F1={score1:.4f}, params={params1}")
    
    if score1 > best_overall_score:
        best_overall_score = score1
        best_overall_params = params1
    
    # PHASE 2: Medium search (30% of trials)
    print("\n" + "="*50)
    print("PHASE 2: Medium search (narrowed range)")
    print("="*50)
    
    # Narrow ranges around Phase 1 best (multiply/divide by 10)
    if model_class.__name__ == "RBFSVM":
        best_g = params1["gamma"]
        best_C = params1["C"]
        
        param_ranges["gamma"] = (best_g / 10, best_g * 10)
        param_ranges["C"] = (best_C / 10, best_C * 10)
        
        print(f"Gamma range: [{best_g/10:.4f}, {best_g*10:.4f}]")
        print(f"C range: [{best_C/10:.2f}, {best_C*10:.2f}]")
    
    # Run phase 2
    n_phase2 = total_trials // 3  # ~10 trials
    print(f"\nTrying {n_phase2} random combinations...")
    params2, score2 = random_search(model_class, X, y, k, n_phase2)
    print(f"Phase 2 best: F1={score2:.4f}, params={params2}")
    
    if score2 > best_overall_score:
        best_overall_score = score2
        best_overall_params = params2
    
    # PHASE 3: Fine search (30% of trials)
    print("\n" + "="*50)
    print("PHASE 3: Fine search (tight range)")
    print("="*50)
    
    # Tighten ranges around Phase 2 best (multiply/divide by 2)
    if model_class.__name__ == "RBFSVM":
        best_g = params2["gamma"]
        best_C = params2["C"]
        
        param_ranges["gamma"] = (best_g / 2, best_g * 2)
        param_ranges["C"] = (best_C / 2, best_C * 2)
        
        print(f"Gamma range: [{best_g/2:.4f}, {best_g*2:.4f}]")
        print(f"C range: [{best_C/2:.2f}, {best_C*2:.2f}]")
    
    # Run phase 3
    n_phase3 = total_trials - n_phase1 - n_phase2  # remaining trials
    print(f"\nTrying {n_phase3} random combinations...")
    params3, score3 = random_search(model_class, X, y, k, n_phase3)
    print(f"Phase 3 best: F1={score3:.4f}, params={params3}")
    
    if score3 > best_overall_score:
        best_overall_score = score3
        best_overall_params = params3
    
    # Restore original ranges
    param_ranges.clear()
    param_ranges.update(original_ranges)
    
    # Final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best F1 score: {best_overall_score:.4f}")
    print(f"Best parameters: {best_overall_params}")
    
    return best_overall_params, best_overall_score
'''