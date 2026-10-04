import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    roc_curve,
    confusion_matrix,
    classification_report
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

# ---------- Random Forest ----------

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_prep, y_train)

y_pred = rf.predict(X_test_prep)
y_proba = rf.predict_proba(X_test_prep)  # columns follow rf.classes_

classes = rf.classes_
is_binary = len(classes) == 2
missing_in_test = sorted(set(classes.tolist()) - set(np.unique(y_test).tolist()))

print("\n========== Random Forest: test metrics ==========")
print("Task:", "binary" if is_binary else f"multiclass ({len(classes)} classes)")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")

if is_binary:
    # Revenue is 0/1, so 1 (made a purchase) is the positive class
    pos_label = 1 if 1 in classes else classes[-1]
    pos_idx = list(classes).index(pos_label)
    y_score = y_proba[:, pos_idx]
    print("Positive class:", pos_label)
    print(f"Precision: {precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0):.4f}")
else:
    for avg in ("macro", "weighted"):
        print(f"Precision ({avg}): {precision_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
        print(f"Recall ({avg}):    {recall_score(y_test, y_pred, average=avg, zero_division=0):.4f}")
        print(f"F1 score ({avg}):  {f1_score(y_test, y_pred, average=avg, zero_division=0):.4f}")

# ROC-AUC needs every class the model knows about to appear in the test labels
roc_auc = None
if missing_in_test:
    print(f"ROC-AUC:   not computed - test set has no samples of class(es) {missing_in_test}, "
          "so the ROC curve is undefined for them")
elif is_binary:
    roc_auc = roc_auc_score(y_test, y_score)
    print(f"ROC-AUC:   {roc_auc:.4f}")
else:
    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=classes)
    print(f"ROC-AUC (macro one-vs-rest): {roc_auc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, labels=classes, zero_division=0))

print("Confusion matrix (rows = true, cols = predicted), labels:", classes.tolist())
print(confusion_matrix(y_test, y_pred, labels=classes))

if is_binary and roc_auc is not None:
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=pos_label)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - Random Forest (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    print("Saved ROC curve to roc_curve.png")
    plt.show()
