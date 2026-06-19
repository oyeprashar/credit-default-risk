from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from src.data.preprocessing import preprocessor, X_train, y_train, X_test, y_test

# Pipeline is a way of grouping things together
# Here pipeline is used to specify how the data is transformed and what model we wanna use
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# accuracy is meaningless when classes are drastically imblanced
# 92% was from class 0, so even randomly predicting class 0 will make it 92% times correct
acc =  np.sum( y_predict == y_test ) / len(y_test)
print("accuracy :", acc)

"""
0 -> No risk 
1 -> Risky Customer 

precision = pr(y=1 | y_hat=1)
recall = pr(y_hat=1 | y=1)

         class   precision  recall  
           0       0.92      1.00     
           1       0.52      0.01 
           
Recall 0.01 for class 1 is terrible. Recall measures the probability of given then the example of class 1, do we predict class 1?   
"""

print(classification_report(y_test, y_predict))

# For every example in X_test, model.predict_proba will find out the probability for it to be an example of 0 or class 1
# two indices at every row! Column 0 is pr to be class 0 and column 1 is the pr to be from class 1
y_predict_prob = model.predict_proba(X_test)

y_prob_class_1 = y_predict_prob[::,1]


# y_test contains the true label
# y_prob_class_1 is probability that the model is predicting 1


"""
# ROC-AUC = Probability that a randomly selected positive example
# receives a higher score than a randomly selected negative example.

# The score here is the model's estimated probability of class 1
# (used later for thresholding and prediction).
"""

print("Precision:", precision_score(y_test, y_predict))
print("Recall:", recall_score(y_test, y_predict))
print("F1:", f1_score(y_test, y_predict))

