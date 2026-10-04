import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

df = pd.read_csv("online_shoppers_intention.csv")

print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# ---------- Preprocessing ----------

# Drop exact duplicate rows so the same session can't land in both train and test
n_before = len(df)
df = df.drop_duplicates()
print(f"Dropped {n_before - len(df)} duplicate rows -> {len(df)} rows")

# Month has a natural order, so encode it as 1-12 (the data spells June out in full)
month_map = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "June": 6, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}
df["Month"] = df["Month"].map(month_map)
assert df["Month"].notna().all(), "Unmapped month value"

# Booleans -> 0/1
df["Weekend"] = df["Weekend"].astype(int)

X = df.drop(columns="Revenue")
y = df["Revenue"].astype(int)

numeric_cols = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "Weekend"
]
# Integer-coded IDs with no real order, plus VisitorType -> one-hot
nominal_cols = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]

# No values are missing today; the imputers are a safeguard.
# Median for numeric because the columns are heavily right-skewed.
preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), nominal_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

print("Train shape after preprocessing:", X_train_prep.shape)
print("Test shape after preprocessing:", X_test_prep.shape)
print("Features:", list(preprocessor.get_feature_names_out()))
print("Revenue rate train/test:", round(y_train.mean(), 3), round(y_test.mean(), 3))