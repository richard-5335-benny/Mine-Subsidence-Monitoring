# IoT-Based Mine Monitoring and Subsidence Risk Prediction System

An ESP32-based prototype for collecting mine-condition sensor readings from multiple monitoring nodes, sending the readings over Wi-Fi to a Python Flask server, and classifying the incoming measurements into **Normal**, **Warning**, or **Critical** risk categories using a trained Random Forest model.

> **Project status:** Prototype / academic engineering project. This system is not a certified mine-safety instrument and must not be used as the sole basis for operational or emergency decisions.

## Contents

- [Overview](#overview)
- [System architecture](#system-architecture)
- [Hardware and software](#hardware-and-software)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Configure the ESP32 nodes](#configure-the-esp32-nodes)
- [Run the Flask monitoring server](#run-the-flask-monitoring-server)
- [HTTP API](#http-api)
- [Machine-learning model](#machine-learning-model)
- [Troubleshooting](#troubleshooting)
- [Limitations and safety](#limitations-and-safety)

## Overview

The project is organized around three ESP32 sensor nodes:

- **N1** – sketch in `circuit_test/`
- **N2** – sketch in `circuit_n2_test/`
- **N3** – sketch in `circuit_n3_test/`

Each node reads vibration, flex/displacement, and IR movement inputs. At a two-second interval, it sends a JSON payload to the Flask server. The server applies the three-feature machine-learning model, stores the latest readings in memory for each node, and serves a web dashboard.

### Main features

- Three separately identified ESP32 monitoring nodes.
- Wi-Fi communication using HTTP POST.
- Flask backend for receiving and processing sensor readings.
- Random Forest classification using vibration, displacement, and movement features.
- Normal, Warning, and Critical class probabilities.
- Dashboard endpoint and latest-node-status endpoint.
- Model-training script and supplied training dataset.

## System architecture

```text
ESP32 Node N1 ─┐
ESP32 Node N2 ─┼── Wi-Fi / HTTP POST ──> Flask Server
ESP32 Node N3 ─┘                           │
                                          ├── Random Forest model
                                          │     └── Risk classification
                                          │
                                          ├── Per-node latest readings
                                          │
                                          └── Web dashboard
```

## Hardware and software

### Hardware

| Component | Purpose |
|---|---|
| ESP32 development boards (up to 3) | Sensor-node controller and Wi-Fi communication |
| Vibration sensor (e.g. SW-420) | Vibration input |
| Flex sensor / flex-sensor circuit | Displacement-related analog input |
| IR sensor | Movement input |
| Jumper wires and suitable power supply | Circuit connections and power |

The exact sensor modules, wiring, calibration, and electrical conditioning should be confirmed against the hardware actually assembled.

### Software

- Arduino IDE with ESP32 board support
- Python 3
- Flask
- NumPy
- pandas
- scikit-learn
- joblib
- A modern web browser

## Repository structure

```text
IoT-Based-Mine-Monitoring/
├── circuit_test/
│   └── circuit_test.ino              # ESP32 node N1
├── circuit_n2_test/
│   └── circuit_n2_test.ino           # ESP32 node N2
├── circuit_n3_test/
│   └── circuit_n3_test.ino           # ESP32 node N3
└── Software/
    ├── app.py                        # Flask server and API
    ├── mine_model_3sensor.pkl        # Model loaded by the server
    ├── mine_subsidence_2node_dataset.csv
    ├── mine_data.csv.xlsx
    ├── predict.py                   # Separate/legacy prediction example
    └── templates/
        ├── dashboard.html            # Monitoring dashboard
        ├── index.html                # Additional HTML page
        ├── train_model.py             # Three-feature model training
        ├── mine_subsidence_training_dataset.csv
        ├── mine_model_3sensor.pkl     # Retained copy of supplied model
        └── mine_model.pkl             # Additional supplied model artifact
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the URL with your GitHub repository URL.

### 2. Create a Python virtual environment

On Ubuntu:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y

cd Software
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you are setting up the project before a `requirements.txt` is present, install the packages with:

```bash
pip install Flask numpy pandas scikit-learn joblib openpyxl
```

## Configure the ESP32 nodes

Open the appropriate `.ino` file in Arduino IDE.

In each sketch, update these configuration values:

```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

const char* SERVER_URL =
  "http://YOUR_COMPUTER_IP:5000/sensor";
```

1. Enter the Wi-Fi network name and password used by the ESP32.
2. Replace `YOUR_COMPUTER_IP` with the LAN IP address of the computer running Flask.
3. Set `NODE_ID` to a unique value: `N1`, `N2`, or `N3`.
4. Check that the GPIO pins match your actual wiring.
5. Select the correct ESP32 board and port in Arduino IDE.
6. Upload the matching sketch to each ESP32.

**Important:** The ESP32 and computer must be able to communicate over the same network. Do not use `127.0.0.1` as the server address in an ESP32 sketch; that address refers to the ESP32 itself. Allow TCP port `5000` through the computer's firewall only on a trusted network, as appropriate.

## Run the Flask monitoring server

The supplied server model is `Software/mine_model_3sensor.pkl`.

From the repository root:

```bash
cd Software
source .venv/bin/activate
python app.py
```

Open the dashboard in a browser:

```text
http://127.0.0.1:5000/
```

For access from another device on the same network, use:

```text
http://YOUR_COMPUTER_IP:5000/
```

The Flask application listens on `0.0.0.0:5000` in the supplied code.

## HTTP API

### `GET /`

Returns the dashboard page.

### `POST /sensor`

Receives a sensor reading and returns the predicted risk level and class probabilities.

Example request:

```json
{
  "node_id": "N1",
  "vibration": 1,
  "flex": 440,
  "movement": 0
}
```

The server accepts `displacement` as an alternative to `flex`.

Example response shape:

```json
{
  "status": "success",
  "node": "N1",
  "risk": "Normal",
  "probabilities": {
    "Normal": 90.0,
    "Warning": 8.0,
    "Critical": 2.0
  }
}
```

The probability values above are illustrative; actual values depend on the loaded model and input.

### `GET /latest`

Returns the latest readings and status for nodes N1, N2, and N3. A node is considered active when it has sent data within the previous five seconds. Node state is held in server memory and is reset when the server restarts.

### `GET /health`

Returns a basic server health response and the configured node identifiers.

## Machine-learning model

The Flask server uses a pre-trained Random Forest classifier with three input features, in this order:

1. `vibration`
2. `displacement`
3. `movement`

The model returns a risk class and class probabilities. The training script is located at `Software/templates/train_model.py` and reads `Software/templates/mine_subsidence_training_dataset.csv`. It splits the dataset into training and test sets, trains a Random Forest classifier, prints evaluation metrics, and saves the resulting model to `Software/mine_model_3sensor.pkl`.

To retrain the model:

```bash
cd Software
source .venv/bin/activate
python templates/train_model.py
```

Retraining overwrites the model file used by the Flask server. Review the printed classification report and validate the model with representative, independently collected data before relying on its predictions.

### Note about `predict.py`

`Software/predict.py` is a separate prediction example that expects a file named `mine_subsidence_model.pkl` and six input features. That model file is not included in this repository, and this script is not the inference path used by `app.py`. The active Flask server uses the three-feature `mine_model_3sensor.pkl` model.

## Troubleshooting

| Problem | What to check |
|---|---|
| `ModuleNotFoundError` | Activate the virtual environment and install `requirements.txt`. |
| Model file not found | Run the server from `Software/` and confirm `mine_model_3sensor.pkl` exists. |
| ESP32 cannot connect | Check Wi-Fi credentials, network access, server IP, firewall, and port `5000`. |
| HTTP connection fails | Confirm Flask is running and the ESP32 URL points to the computer's LAN IP. |
| Invalid node response | Use only `N1`, `N2`, or `N3` for `node_id`. |
| Dashboard shows no recent data | Confirm the ESP32 is sending POST requests successfully. |

## Limitations and safety

- This is a prototype and is not certified for mine monitoring or worker protection.
- The quality of predictions depends on the training dataset, sensor calibration, installation, and environmental conditions.
- Validate the model using representative field measurements and suitable engineering review.
- The dashboard's latest node data is stored in memory; there is no persistent database in the supplied backend.
- The example HTTP interface does not implement authentication or encrypted transport. Do not expose it directly to the public internet.
- Pickle/joblib model files can execute code when loaded. Only load model files from sources you trust.

## License

No license was supplied with the original project. Add a license only after deciding how you want others to use, modify, and distribute this work.
