from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from src.data.preprocessing import preprocessor, X_train, y_train, X_test, y_test
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

"""
92 % example are of class 0 
8 % examples are of class 1

Even if we predict class 0 blindly, the accuracy is still 92%

ratio = (negative_examples / positive_examples) = 11%

This ratio says that making a mistake over class 1 should be 11 times more expensive 
than making a mistake over class 0 since they are imbalanced

"""

negative_examples = np.sum(y_train == 0)
positive_examples = np.sum(y_train == 1)
ratio = (negative_examples / positive_examples)

# now we need to create the model st we give it this preprocessor
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
    n_estimators=100, # XGBoost is boosting and uses weak learners
    max_depth=6, # depth of the decision trees being used
    learning_rate=0.1,
    random_state=1234,
    scale_pos_weight=ratio, )),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = (np.sum(y_pred == y_test))/len(y_test)
# print("accuracy :", acc)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

y_prob = model.predict_proba(X_test)
y_predit_prob_class_1 = y_prob[::,1]

# we need to find out the roc-auc score for class 1

print("ROC-AUC:", roc_auc_score(y_test, y_predit_prob_class_1))