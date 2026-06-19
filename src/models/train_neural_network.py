import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.data.preprocessing import preprocessor, X_train, y_train, X_test, y_test
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(128, 64), # Layer 1: 128 neurons  | # Layer 2: 64 neurons
        activation="relu",
        max_iter=100,
        random_state=1234)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = (np.sum( y_pred == y_test)) / len(y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
y_prob = model.predict_proba(X_test)
y_predit_prob_class_1 = y_prob[::,1]
# we need to find out the roc-auc score for class 1
print("ROC-AUC:", roc_auc_score(y_test, y_predit_prob_class_1))