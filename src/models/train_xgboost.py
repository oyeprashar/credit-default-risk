import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
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


# now we need to separate feature and label
X = df.drop(columns=["TARGET"])
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234, test_size = 0.2, stratify=y)

numerical_col_names = X_train.select_dtypes(
    include = ["int64", "float64"]
).columns

categorical_col_names = X_train.select_dtypes(
    include = ["object"]
).columns


# now we need to define the transformers
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # handle the missing NaN values with median
    ("scaler", StandardScaler()) # if age was 15, 17, 20, 23, 25 it will become -2, -1, 0, 1, 2
])

# transformer for categorical data
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # the NaN field is filled with most frequent category
    ("encoder", OneHotEncoder(handle_unknown="ignore")) # OneHotEncoder converts categorical features into binary indicator columns.
])


# now we need to map transformer and col names
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_col_names),
    ("cat", categorical_transformer, categorical_col_names)
])

# now we need to create the model st we give it this preprocessor
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
    n_estimators=100, # XGBoost is boosting and uses weak learners
    max_depth=6, # depth of the decision trees being used
    learning_rate=0.1,
    random_state=1234)),
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
