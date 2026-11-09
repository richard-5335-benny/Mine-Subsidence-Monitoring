import joblib
import pandas as pd

# Load trained model
model = joblib.load("mine_subsidence_model.pkl")

# ==========================================
# ENTER SENSOR VALUES
# ==========================================

vibration = 85
flex = 440
ir = 1
deformation = 11.5
frequency = 25
risk_score = 92

# ==========================================
# CREATE INPUT
# ==========================================

data = pd.DataFrame([{
    "vibration_level": vibration,
    "flex_value": flex,
    "ir_motion": ir,
    "deformation_mm": deformation,
    "vibration_frequency": frequency,
    "risk_score": risk_score
}])

# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(data)[0]

probability = model.predict_proba(data)[0]

print("\n===================================")
print("MINE SUBSIDENCE PREDICTION")
print("===================================")

print("Risk Level:", prediction)

print("\nPrediction Probability:")

for label, prob in zip(model.classes_, probability):
    print(label, ":", round(prob * 100, 2), "%")
