import pandas as pd
from scipy.constants import precision
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

base_path = "/Users/shubhamprashar/Desktop/Interview Prep/ml_projects/credit-default-risk/data"
df = pd.read_csv(base_path + "/application_train.csv")

# now we need to separate the features and target
X = df.drop(columns = ["TARGET"])
y = df["TARGET"]


# stratify=y makes sure that the split contains the original ratio of +ve and -ve examples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)

# *--------------------- Pre-processing ---------------------*
# Logistic regression can only understand numerical data
# We need to convert categorical data into numerical

### Get the column names for numerical and categorical (no data, just the column names)
numerical_cols = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns


numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # handle the missing NaN values with median
    ("scaler", StandardScaler()) # if age was 15, 17, 20, 23, 25 it will become -2, -1, 0, 1, 2

])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # the NaN field is filled with most frequent category
    ("encoder", OneHotEncoder(handle_unknown="ignore")) # OneHotEncoder converts categorical features into binary indicator columns.
])


# For all columns listed in numerical_cols, apply numeric_transformer.
# We are not doing any transformation just creating configuration for the transformation
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])


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

