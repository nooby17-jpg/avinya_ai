import json
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --------------------------------
# PATHS
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(
    BASE_DIR, "..", "data", "simulation", "avinya_simulation_data.json"
)
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "emergency_lstm.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

# --------------------------------
# LOAD DATA
# --------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

patients = data["patients"]
texts = [p["reported_symptom"] for p in patients]

labels = [
    0 if p["severity_level"] == "low"
    else 1 if p["severity_level"] == "medium"
    else 2
    for p in patients
]

print(f"✅ Loaded {len(texts)} samples")

# --------------------------------
# TOKENIZATION
# --------------------------------
VOCAB_SIZE = 5000
MAX_LEN = 10

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN)

labels_cat = to_categorical(labels, num_classes=3)

# --------------------------------
# TRAIN / TEST SPLIT
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels_cat, test_size=0.2, random_state=42
)

# --------------------------------
# BUILD LSTM MODEL
# --------------------------------
model = Sequential([
    Embedding(VOCAB_SIZE, 64, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------
# TRAIN
# --------------------------------
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

# --------------------------------
# SAVE MODEL + TOKENIZER
# --------------------------------
model.save(MODEL_PATH)

with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ LSTM model saved")
