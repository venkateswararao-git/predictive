import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

DATA_FILE = "C:/Users/harip/OneDrive/Desktop/air quality/air_quality_dataset (2).csv"
OUTPUT_DIR = "air_quality_project_outputs"
TEST_SIZE = 0.20
RANDOM_STATE = 42
TARGET = "AIR_QUALITY"
SOURCE_COLUMN = "PM2.5"
CLASS_ORDER = ["Good", "Satisfactory", "Moderate", "Poor"]

PM25_BINS = [-np.inf, 30, 60, 90, np.inf]

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. Put the CSV file in the same folder as this script "
        "or change DATA_FILE to the full path of your dataset."
    )

try:
    data = pd.read_csv(DATA_FILE)
except UnicodeDecodeError:
    data = pd.read_csv(DATA_FILE, encoding="latin-1")
data.columns = data.columns.str.strip()

required_columns = ["PM2.5", "PM10", "NOx", "NO2", "SO2", "VOCs", "CO", "CO2", "CH4", "Temperature"]
missing_columns = [c for c in required_columns if c not in data.columns]
if missing_columns:
    raise ValueError(
        "Missing expected columns: " + ", ".join(missing_columns) +
        "\nColumns found: " + ", ".join(map(str, data.columns))
    )

print("=" * 60)
print("PREDICTIVE ANALYTICS FOR AIR QUALITY CATEGORY")
print("=" * 60)
print("\nDataset shape:", data.shape)
print("\nFirst five rows:\n", data.head())
print("\nDataset information:")
data.info()
print("\nDescriptive statistics:\n", data.describe().transpose())
print("\nMissing values:\n", data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())
print("\nNegative readings per column:\n", (data < 0).sum()[lambda s: s > 0])

data.describe().transpose().to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"))
data.isnull().sum().rename("missing_values").to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))

data = data.drop_duplicates().copy()
data[TARGET] = pd.cut(data[SOURCE_COLUMN], bins=PM25_BINS, labels=CLASS_ORDER).astype(str)
print("\nAir-quality class counts:\n", data[TARGET].value_counts().reindex(CLASS_ORDER))
data[TARGET].value_counts().reindex(CLASS_ORDER).rename("count").to_csv(
    os.path.join(OUTPUT_DIR, "air_quality_class_counts.csv"))

leakage_cols = [SOURCE_COLUMN, TARGET]
print("\nColumns excluded from the model inputs:", leakage_cols)

numeric_features = ["PM10", "NOx", "NO2", "SO2", "VOCs", "CO", "CO2", "CH4", "Temperature"]
numeric_columns = ["PM2.5"] + numeric_features

class_labels = [c for c in CLASS_ORDER if c in data[TARGET].unique()]

plt.figure(figsize=(10, 3.3))
class_counts = data[TARGET].value_counts().reindex(class_labels)
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.title("Distribution of Air-Quality Classes")
plt.xlabel("Air-quality class")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=200)
plt.show()
plt.close()

data[numeric_columns].hist(figsize=(10, 6.0), bins=20, layout=(3, 4))
plt.suptitle("Distributions of Numerical Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "numeric_distributions.png"), dpi=200)
plt.show()
plt.close("all")

corr = data[numeric_columns].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 4.7))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=200)
plt.show()
plt.close(fig)

boxplot_groups = [["PM10"], ["NOx", "NO2"], ["SO2"], ["VOCs"], ["CO"], ["CO2"], ["CH4", "Temperature"]]
for group in boxplot_groups:
    fig, axes = plt.subplots(1, len(group), figsize=(10, 2.85), squeeze=False)
    for ax, feature in zip(axes[0], group):
        values = [data.loc[data[TARGET] == r, feature].dropna().values for r in class_labels]
        ax.boxplot(values, showfliers=True)
        ax.set_xticklabels(class_labels)
        ax.set_title(f"{feature} Across Air-Quality Classes")
        ax.set_xlabel("Air-quality class")
        ax.set_ylabel(feature)
    fig.tight_layout()
    safe_name = "_".join(group).lower()
    fig.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{safe_name}.png"), dpi=200)
    plt.show()
    plt.close(fig)

z = (data[numeric_features] - data[numeric_features].mean()) / data[numeric_features].std()
class_means = z.groupby(data[TARGET]).mean().loc[class_labels]
fig, ax = plt.subplots(figsize=(10, 3.4))
im = ax.imshow(class_means.values, cmap="coolwarm", aspect="auto")
fig.colorbar(im, ax=ax, label="Mean z-score")
ax.set_xticks(range(len(class_means.columns)))
ax.set_xticklabels(class_means.columns, rotation=45, ha="right")
ax.set_yticks(range(len(class_means.index)))
ax.set_yticklabels(class_means.index)
for i in range(class_means.shape[0]):
    for j in range(class_means.shape[1]):
        ax.text(j, i, f"{class_means.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Class-wise Mean of Standardised Features")
ax.set_xlabel("Feature")
ax.set_ylabel("Air-quality class")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "class_feature_heatmap.png"), dpi=200)
plt.show()
plt.close(fig)
class_means.to_csv(os.path.join(OUTPUT_DIR, "class_feature_means.csv"))

X = data[numeric_features].copy()
y = data[TARGET].copy()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features)
])

stratify_target = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_target
)
print("\nTrain size:", X_train.shape[0], "| Test size:", X_test.shape[0])

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
}

results = []
labels = class_labels

for model_name, classifier in models.items():
    print("\n" + "=" * 60 + "\n" + model_name + "\n" + "=" * 60)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    file_stub = model_name.lower().replace(" ", "_")
    pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    ).to_csv(os.path.join(OUTPUT_DIR, f"{file_stub}_confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Number of records")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Air-Quality Class")
    ax.set_ylabel("Actual Air-Quality Class")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(label) for label in labels])
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{file_stub}_confusion_matrix.png"), dpi=200)
    plt.show()
    plt.close(fig)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print("\n" + "=" * 60 + "\nFINAL MODEL COMPARISON\n" + "=" * 60)
print(results_df.round(4).to_string(index=False))
print("\nOutput files saved in:", os.path.abspath(OUTPUT_DIR))
