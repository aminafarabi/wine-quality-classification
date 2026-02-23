from data_loader import load_wine_data
from models.svm import SVM
from model_selection import random_search
from model_selection import train_test_split
from model_selection import accuracy, precision, f1_score, recall
import numpy as np

X, y = load_wine_data(
    "data/winequality-red.csv",
    "data/winequality-white.csv"
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test - mean) / std


print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


param_grid = {
    "learning_rate": [0.0001, 0.001],
    "lambda_param": [0.001, 0.01],
    "n_iters": [500, 1000]
}

# best_params, best_score = grid_search(SVM, param_grid, X, y, k=5)
best_params, best_score = random_search(SVM, X_train_scaled, y_train, k=5)

print("Best params:", best_params)
# print("Best CV accuracy:", best_score)

final_model = SVM(**best_params)
final_model.fit(X_train_scaled, y_train)

y_pred_test = final_model.predict(X_test_scaled)

test_acc = accuracy(y_test, y_pred_test)
test_recall = recall(y_test, y_pred_test)
test_prec = precision(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print("Final test accuracy:", test_acc)
print("Final test precision:", test_prec)
print("Final test recall:", test_recall)
print("Final test f1 score:", test_f1)


