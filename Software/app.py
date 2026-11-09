from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import time
from pathlib import Path

app = Flask(__name__)

# ==========================================================
# LOAD ML MODEL
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "mine_model_3sensor.pkl"

model = joblib.load(MODEL_FILE)

print("================================")
print("     MINE MONITORING SERVER")
print("================================")
print("ML Model Loaded Successfully")
print()


# ==========================================================
# NODE DATA
# ==========================================================

nodes = {

    "N1": {
        "vibration": 0.0,
        "displacement": 0.0,
        "movement": 0.0,

        "risk": "WAITING",

        "probabilities": {
            "Normal": 0.0,
            "Warning": 0.0,
            "Critical": 0.0
        },

        "active": False,
        "last_seen": 0
    },

    "N2": {
        "vibration": 0.0,
        "displacement": 0.0,
        "movement": 0.0,

        "risk": "WAITING",

        "probabilities": {
            "Normal": 0.0,
            "Warning": 0.0,
            "Critical": 0.0
        },

        "active": False,
        "last_seen": 0
    },

    "N3": {
        "vibration": 0.0,
        "displacement": 0.0,
        "movement": 0.0,

        "risk": "WAITING",

        "probabilities": {
            "Normal": 0.0,
            "Warning": 0.0,
            "Critical": 0.0
        },

        "active": False,
        "last_seen": 0
    }
}


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/")
def dashboard():

    return render_template("dashboard.html")


# ==========================================================
# RECEIVE SENSOR DATA
# ==========================================================

@app.route("/sensor", methods=["POST"])
def sensor():

    try:

        data = request.get_json()

        # --------------------------------------------------
        # NODE
        # --------------------------------------------------

        node = data.get("node_id", "N1")

        if node not in nodes:

            return jsonify({
                "status": "error",
                "message": "Invalid node"
            }), 400


        # --------------------------------------------------
        # SENSOR VALUES
        # --------------------------------------------------

        vibration = float(
            data.get("vibration", 0)
        )

        displacement = float(
            data.get(
                "displacement",
                data.get("flex", 0)
            )
        )

        movement = float(
            data.get("movement", 0)
        )


        # --------------------------------------------------
        # PRINT RECEIVED DATA
        # --------------------------------------------------

        print()
        print("==============================")
        print("RECEIVED SENSOR DATA")
        print("==============================")

        print("Node:", node)
        print("Vibration:", vibration)
        print("Displacement:", displacement)
        print("Movement:", movement)


        # --------------------------------------------------
        # ML PREDICTION
        # --------------------------------------------------

        features = np.array([[
            vibration,
            displacement,
            movement
        ]])


        prediction = model.predict(features)[0]

        probabilities_array = (
            model.predict_proba(features)[0]
        )


        # --------------------------------------------------
        # CONVERT PREDICTION TO STRING
        # --------------------------------------------------

        risk = str(prediction)

        risk = risk.capitalize()


        # --------------------------------------------------
        # PROBABILITIES
        # --------------------------------------------------

        probabilities = {

            "Normal": 0.0,
            "Warning": 0.0,
            "Critical": 0.0

        }


        for class_name, probability in zip(
            model.classes_,
            probabilities_array
        ):

            class_name = str(class_name).capitalize()

            if class_name in probabilities:

                probabilities[class_name] = round(
                    float(probability) * 100,
                    1
                )


        # --------------------------------------------------
        # PRINT ML RESULT
        # --------------------------------------------------

        print()
        print("==============================")
        print("ML PREDICTION")
        print("==============================")

        print("Node:", node)
        print("Risk Level:", risk)

        print()
        print("Prediction Probability:")

        print(
            "Normal:",
            probabilities["Normal"],
            "%"
        )

        print(
            "Warning:",
            probabilities["Warning"],
            "%"
        )

        print(
            "Critical:",
            probabilities["Critical"],
            "%"
        )


        # ==================================================
        # UPDATE ONLY THIS NODE
        # ==================================================

        nodes[node] = {

            "vibration": vibration,

            "displacement": displacement,

            "movement": movement,

            "risk": risk,

            "probabilities": probabilities,

            "active": True,

            "last_seen": time.time()
        }


        # ==================================================
        # RESPONSE
        # ==================================================

        return jsonify({

            "status": "success",

            "node": node,

            "risk": risk,

            "probabilities": probabilities

        })


    except Exception as e:

        print()
        print("ERROR:")
        print(e)

        return jsonify({

            "status": "error",
            "message": str(e)

        }), 500


# ==========================================================
# LATEST DATA
# ==========================================================

@app.route("/latest", methods=["GET"])
def latest():

    current_time = time.time()


    # ------------------------------------------------------
    # UPDATE ONLINE/OFFLINE STATUS INDEPENDENTLY
    # ------------------------------------------------------

    output = {}


    for node_name, node_data in nodes.items():

        node_copy = node_data.copy()


        # Each node has its own status
        if (
            current_time -
            node_data["last_seen"]
            <= 5
        ):

            node_copy["active"] = True

        else:

            node_copy["active"] = False


        output[node_name] = node_copy


    # ------------------------------------------------------
    # RETURN ALL NODES
    # ------------------------------------------------------

    return jsonify(output)


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "running",

        "nodes": list(nodes.keys())

    })


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    print()
    print("================================")
    print("Server starting...")
    print("Dashboard:")
    print("http://0.0.0.0:5000")
    print("================================")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
