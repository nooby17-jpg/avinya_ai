import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR, "..", "data", "simulation", "avinya_simulation_data.json"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

patients = data["patients"]
print(f"✅ Loaded {len(patients)} patient records")

texts = [p["reported_symptom"] for p in patients]
labels = [
    0 if p["severity_level"] == "low"
    else 1 if p["severity_level"] == "medium"
    else 2
    for p in patients
]

# -------------------------------
# TRAIN / TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -------------------------------
# TRAIN MODEL
# -------------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -------------------------------
# EVALUATE
# -------------------------------
print("\n📊 Classification Report\n")
print(classification_report(y_test, model.predict(X_test_vec)))

# -------------------------------
# SAVE MODEL
# -------------------------------
joblib.dump(model, os.path.join(MODEL_DIR, "emergency_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))

print("✅ ML model saved successfully")
