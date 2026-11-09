import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CSV_FILE = SCRIPT_DIR / "mine_subsidence_training_dataset.csv"

df = pd.read_csv(CSV_FILE)

print("\n================================")
print("       DATASET INFORMATION")
print("================================")

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())


# ==========================================
# 2. CREATE MOVEMENT FEATURE
# ==========================================

# Use displacement_rate to estimate movement.
#
# Adjust this threshold later using real
# measurements from your IR sensor.

MOVEMENT_THRESHOLD = 0.20

df["movement"] = (
    df["displacement_rate"] > MOVEMENT_THRESHOLD
).astype(int)


# ==========================================
# 3. SELECT 3 FEATURES
# ==========================================

features = [
    "vibration",
    "displacement",
    "movement"
]

target = "risk_level"


X = df[features]
y = df[target]


# ==========================================
# 4. SHOW CLASS DISTRIBUTION
# ==========================================

print("\n================================")
print("       CLASS DISTRIBUTION")
print("================================")

print(y.value_counts())

print("\nPercentages:")
print(
    y.value_counts(normalize=True) * 100
)


# ==========================================
# 5. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ==========================================
# 6. CREATE RANDOM FOREST
# ==========================================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    class_weight="balanced",

    n_jobs=-1
)


# ==========================================
# 7. TRAIN MODEL
# ==========================================

print("\n================================")
print("       TRAINING MODEL")
print("================================")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ==========================================
# 8. TEST MODEL
# ==========================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n================================")
print("       MODEL RESULTS")
print("================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 9. SAVE MODEL
# ==========================================

MODEL_FILE = PROJECT_DIR / "mine_model_3sensor.pkl"

joblib.dump(
    model,
    MODEL_FILE
)


print("\n================================")
print("       MODEL SAVED")
print("================================")

print(
    "File:",
    MODEL_FILE
)

print("\nFeatures used:")

for feature in model.feature_names_in_:

    print(
        "-",
        feature
    )

print("\nClasses:")

print(
    model.classes_
)

print("\nDONE!")
