from data_loader import load_wine_data
from models.kernel import Kernel_SVM
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


svm = Kernel_SVM(kernel="rbf", C=1.0, gamma=0.1)
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)

test_acc = accuracy(y_test, y_pred)
test_recall = recall(y_test, y_pred)
test_prec = precision(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("Final test accuracy:", test_acc)
print("Final test precision:", test_prec)
print("Final test recall:", test_recall)
print("Final test f1 score:", test_f1)