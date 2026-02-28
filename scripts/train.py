from data_loader import load_wine_data
from models.svm import LinearSVM
from models.kernel import RBFSVM
from models.log_reg import LogisticRegression
from model_selection import random_search
from model_selection import train_test_split
from model_selection import accuracy, precision, f1_score, recall
import numpy as np

X, y = load_wine_data(
    "data/winequality-red.csv",
    "data/winequality-white.csv"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
train_std = np.where(train_std == 0, 1, train_std)  # avoiding division by zero
X_train_scaled = (X_train - train_mean) / train_std
X_test_scaled = (X_test - train_mean) / train_std

results = []
models_list = [LinearSVM, RBFSVM, LogisticRegression]
for model in models_list:
    best_params, best_score = random_search(model, X_train, y_train, k=5, n_trials=30)

    final_model = model(**best_params)
    final_model.fit(X_train_scaled, y_train)

    y_pred_test = final_model.predict(X_test_scaled)

    test_acc = accuracy(y_test, y_pred_test)
    test_recall = recall(y_test, y_pred_test)
    test_prec = precision(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    result = {
        "model": model.__name__,
        "accuracy": test_acc,
        "recall": test_recall,
        "precision": test_prec,
        "f1_score": test_f1
    }
    print(result)
    results.append(result)

# After the main loop, print all results
print("\n" + "="*50)
print("FINAL RESULTS FOR ALL MODELS")
print("="*50)

for result in results:
    print(f"\nModel: {result['model']}")
    print("-" * 30)
    for metric, value in result.items():
        if metric != "model":
            print(f"{metric.capitalize():15}: {value:.4f}")