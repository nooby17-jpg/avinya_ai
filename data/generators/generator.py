import json
import random
import os
from datetime import datetime, timedelta

# -------------------------------
# CONFIGURATION
# -------------------------------
NUM_PATIENTS = 10000
OUTPUT_FILE = "avinya_simulation_data.json"

first_names = [
    "Pema", "Tashi", "Anjali", "Sanjay", "Kiran", "Sunita",
    "Deekila", "Nima", "Ongdi", "Nikhil", "Sanidhya",
    "Ashish", "Rahul", "Dorjee", "Binod", "Phurba",
    "Rajesh", "Mingma", "Karma", "Shiban", "Sangeeta",
    "Rinchen", "Sarad", "Razal", "Sasank"
]

last_names = [
    "Lepcha", "Bhutia", "Tamang", "Pradhan",
    "Gurung", "Rai", "Chettri", "Sharma", "Subba", "Wangdi"
]

symptom_pool = [
    ("fever", "low"),
    ("cold and cough", "low"),
    ("headache", "medium"),
    ("abdominal pain", "medium"),
    ("fracture pain", "medium"),
    ("chest pain", "high"),
    ("breathlessness", "high"),
    ("severe bleeding", "high"),
    ("loss of consciousness", "high")
]

department_map = {
    "low": "General Medicine",
    "medium": "OPD",
    "high": "Emergency"
}

# -------------------------------
# GENERATE PATIENTS
# -------------------------------
patients = []

for i in range(1, NUM_PATIENTS + 1):
    symptom, severity = random.choice(symptom_pool)

    patients.append({
        "patient_id": f"SIM_PAT_{i:06d}",
        "age": random.randint(1, 90),
        "gender": random.choice(["Male", "Female"]),
        "reported_symptom": symptom,
        "severity_level": severity,
        "assigned_department": department_map[severity],
        "is_emergency": severity == "high",
        "timestamp": (
            datetime.now() - timedelta(
                minutes=random.randint(1, 200000)
            )
        ).isoformat()
    })

# -------------------------------
# FINAL JSON
# -------------------------------
output = {
    "simulation_info": {
        "hospital_name": "Avinya Hospital",
        "synthetic": True,
        "patient_count": NUM_PATIENTS,
        "created_for": "Emergency risk classification using ML/DL",
        "data_usage": "Training and evaluation only"
    },
    "label_mapping": {
        "low": 0,
        "medium": 1,
        "high": 2
    },
    "patients": patients
}

# -------------------------------
# SAVE
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, OUTPUT_FILE), "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4)

print("✅ Simulation data generated")
print(f"👥 Patients: {NUM_PATIENTS}")
print(f"📁 File: {OUTPUT_FILE}")
