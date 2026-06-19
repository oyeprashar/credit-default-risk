import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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