import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

base_path = "/Users/shubhamprashar/Desktop/Interview Prep/ml_projects/credit-default-risk/data"
df = pd.read_csv(base_path + "/application_train.csv")

# now we separate the example and labels

X = df.drop(columns=["TARGET"])
y = df["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234, stratify=y)


# numerical col names
numerical_col_names = X_train.select_dtypes(
    include=["int64", "float64"]
).columns


# categorical col names
categorical_col_names = X_train.select_dtypes(
    include=["object"]
).columns


# These are specific transformation config for each kind of cols
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")), # handle the missing NaN values with median
    ("scaler", StandardScaler()) # if age was 15, 17, 20, 23, 25 it will become -2, -1, 0, 1, 2

])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")), # the NaN field is filled with most frequent category
    ("encoder", OneHotEncoder(handle_unknown="ignore")) # OneHotEncoder converts categorical features into binary indicator columns.
])


# now we map the col names to the transformer that will be applied
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_col_names),
    ("cat", categorical_transformer, categorical_col_names)
])

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
y_predict_prob =  model.predict_proba(X_test)
y_predit_prob_class_1 = y_predict_prob[::,1]


acc = (np.sum(y_predict == y_test))/len(y_test)
print("Accuracy:", accuracy_score(y_test, y_predict))
print("Precision:", precision_score(y_test, y_predict))
print("Recall:", recall_score(y_test, y_predict))
print("F1:", f1_score(y_test, y_predict))

# We need to analyze the score

print("ROC-AUC:", roc_auc_score(y_test, y_predit_prob_class_1))
