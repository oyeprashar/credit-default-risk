import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
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
    ("classifier", RandomForestClassifier (
        n_estimators=100, # number of decision trees in random forest
        random_state=1234,
        n_jobs=-1)) #  controls parallelism — it tells sklearn to use all available CPU cores to train the trees in parallel, speeding up fitting.
])

model.fit(X_train, y_train)
y_predict = model.predict(X_test)


# this will give us the probability of an example
# belonging to class 0 and class 1


acc = (np.sum(y_predict == y_test))/len(y_test)
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Precision:", precision_score(y_test, y_predict))
print("Recall:", recall_score(y_test, y_predict))
print("F1:", f1_score(y_test, y_predict))


y_predict_prob =  model.predict_proba(X_test)
y_predit_prob_class_1 = y_predict_prob[::,1]
print("ROC-AUC:", roc_auc_score(y_test, y_predit_prob_class_1))
