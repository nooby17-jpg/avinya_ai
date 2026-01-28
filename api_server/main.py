# =========================================================
# IMPORTS
# =========================================================
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import joblib
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(
    title="AVYA - AI Hospital Receptionist API",
    description="AI-powered hospital receptionist and emergency assistance system",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)


# =========================================================
# LOAD HOSPITAL CONFIG (LOCKED PER HOSPITAL)
# =========================================================
HOSPITAL_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "hospital_configs",
    "avinya.json"
)

if not os.path.exists(HOSPITAL_CONFIG_PATH):
    raise RuntimeError("Hospital configuration file not found")

with open(HOSPITAL_CONFIG_PATH, "r", encoding="utf-8") as f:
    hospital_config = json.load(f)


# =========================================================
# LOAD CLASSICAL ML MODEL
# =========================================================
MODEL_DIR = os.path.join(PROJECT_ROOT, "core_ai", "models")

ML_MODEL_PATH = os.path.join(MODEL_DIR, "emergency_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

if not os.path.exists(ML_MODEL_PATH):
    raise RuntimeError("ML emergency model not found")

emergency_ml_model = joblib.load(ML_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


# =========================================================
# LOAD DEEP LEARNING (LSTM) MODEL
# =========================================================
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "emergency_lstm.h5")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")

if not os.path.exists(LSTM_MODEL_PATH):
    raise RuntimeError("LSTM model not found")

lstm_model = load_model(LSTM_MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 10


# =========================================================
# REQUEST MODELS
# =========================================================
class ChatRequest(BaseModel):
    message: str


class EmergencyCheckRequest(BaseModel):
    symptom_text: str


# =========================================================
# DEEP LEARNING (LSTM) INFERENCE
# =========================================================
def dl_detect_emergency(text: str) -> dict:
    seq = tokenizer.texts_to_sequences([text.lower()])
    padded = pad_sequences(seq, maxlen=MAX_LEN)

    probs = lstm_model.predict(padded, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "model": "LSTM"
    }


# =========================================================
# CLASSICAL ML INFERENCE
# =========================================================
def ml_detect_emergency(text: str) -> dict:
    vec = vectorizer.transform([text])
    probs = emergency_ml_model.predict_proba(vec)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "model": "LogisticRegression"
    }


# =========================================================
# HYBRID DECISION ENGINE (DL + ML + SAFETY)
# =========================================================
def hybrid_emergency_decision(text: str) -> dict:
    safety_cfg = hospital_config.get("ai_safety", {})
    threshold = safety_cfg.get("emergency_confidence_threshold", 0.75)
    force_symptoms = safety_cfg.get("force_emergency_symptoms", [])

    text_lower = text.lower()

    # Rule-based safety override
    for symptom in force_symptoms:
        if symptom in text_lower:
            return {
                "risk_level": "EMERGENCY",
                "confidence": 1.0,
                "decision_reason": "Rule-based safety override",
                "recommended_action": "Proceed to Emergency Department immediately",
                "emergency_number": hospital_config["hospital_metadata"]["emergency_number"]
            }

    # Deep Learning decision
    dl_result = dl_detect_emergency(text)

    if dl_result["predicted_class"] == 2 and dl_result["confidence"] >= threshold:
        return {
            "risk_level": "EMERGENCY",
            "confidence": dl_result["confidence"],
            "decision_reason": "LSTM confidence above threshold",
            "recommended_action": "Proceed to Emergency Department immediately",
            "emergency_number": hospital_config["hospital_metadata"]["emergency_number"]
        }

    if dl_result["predicted_class"] == 1:
        return {
            "risk_level": "MEDIUM_RISK",
            "confidence": dl_result["confidence"],
            "decision_reason": "LSTM medium-risk classification",
            "recommended_action": "Consult doctor soon"
        }

    # Fallback to classical ML
    ml_result = ml_detect_emergency(text)

    if ml_result["predicted_class"] == 2:
        return {
            "risk_level": "EMERGENCY",
            "confidence": ml_result["confidence"],
            "decision_reason": "ML fallback emergency detection",
            "recommended_action": "Proceed to Emergency Department immediately",
            "emergency_number": hospital_config["hospital_metadata"]["emergency_number"]
        }

    return {
        "risk_level": "NON_EMERGENCY",
        "confidence": max(dl_result["confidence"], ml_result["confidence"]),
        "decision_reason": "Low-risk classification",
        "recommended_action": "OPD consultation suggested"
    }


# =========================================================
# API ENDPOINTS
# =========================================================
@app.get("/")
def root():
    return {
        "assistant": "AVYA",
        "hospital": hospital_config["hospital_metadata"]["name"],
        "status": "AI Receptionist Online"
    }


@app.post("/chat")
def chat(req: ChatRequest):
    result = hybrid_emergency_decision(req.message)

    if result["risk_level"] == "EMERGENCY":
        return {"type": "emergency", "response": result}

    if "doctor" in req.message.lower():
        return {"type": "info", "response": hospital_config["staff"]["doctors"]}

    if "hospital" in req.message.lower():
        return {"type": "info", "response": hospital_config["hospital_metadata"]}

    return {
        "type": "general",
        "response": (
            f"Hello, I am AVYA, the AI receptionist of "
            f"{hospital_config['hospital_metadata']['name']}. "
            "How may I assist you?"
        )
    }


@app.post("/emergency-check")
def emergency_check(req: EmergencyCheckRequest):
    return hybrid_emergency_decision(req.symptom_text)


@app.get("/hospital-info")
def hospital_info():
    return hospital_config["hospital_metadata"]


@app.get("/doctors")
def get_doctors():
    return hospital_config["staff"]["doctors"]


@app.get("/health")
def health():
    return {
        "status": "OK",
        "hospital": hospital_config["hospital_metadata"]["name"]
    }
