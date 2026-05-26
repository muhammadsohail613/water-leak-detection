<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=32&duration=3000&pause=1000&color=00D4FF&center=true&vCenter=true&width=600&lines=💧+AquaGuard+AI;Water+Leak+Detection+System;Real-Time+%7C+AI-Powered+%7C+Ultrasonic" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.12-00d4ff?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-f7931e?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-3d4db7?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00cc66?style=for-the-badge)

<br/>

> **AI-powered water leak detection system using Isolation Forest anomaly detection on ultrasonic flow meter data — with a real-time live monitoring dashboard.**

<br/>

🎥 **[Watch Live Demo](https://www.loom.com/share/d16b4247e17e4ef9aa2915f57a33b2fb)**

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Use Cases](#-use-cases)
- [Author](#-author)

---

## 🔍 Overview

**AquaGuard AI** is an end-to-end machine learning system designed to detect water pipe leaks in real time using data from **ultrasonic flow meters**. It monitors three key signals — **flow rate**, **pressure**, and **velocity** — and uses an **Isolation Forest** anomaly detection model to flag leaks the moment they occur, with severity classification (CRITICAL / HIGH / MEDIUM).

The system ships with a **live streaming dashboard** built in Streamlit, where leaks can be monitored continuously, triggered manually for testing, and analyzed with interactive charts.

---

## 🎥 Live Demo

> Click the thumbnail below to watch the full demo:

[![Watch Demo](https://img.shields.io/badge/▶%20Watch%20Demo-Loom-00d4ff?style=for-the-badge&logo=loom&logoColor=white)](https://www.loom.com/share/d16b4247e17e4ef9aa2915f57a33b2fb)

**Demo covers:**
- 📡 Real-time sensor data streaming
- 🚨 Live leak detection with instant alerts
- 💥 Manual leak injection simulation
- 📊 Multi-chart live dashboard
- 🔍 Custom sensor reading analysis via sidebar

---

## ✨ Features

### 📊 Historical Dashboard
- Full dataset analysis with 1,000+ flow meter readings
- Interactive flow rate, pressure, and anomaly score charts
- Severity breakdown pie chart (CRITICAL / HIGH / MEDIUM)
- Filterable data table by severity and leak status
- Manual prediction panel — enter any sensor values and get instant AI verdict

### 📡 Live Monitor
| Feature | Details |
|---|---|
| ▶ Start / ⏹ Stop | Full control of the live data stream |
| 💥 Inject Leak | Manually trigger a leak event on demand |
| ⚡ Speed Control | Adjustable update rate: 0.3s → 2s per reading |
| 🔴 Live Status Banner | Pulses red on leak, green on normal flow |
| 📈 4 Live Charts | Flow rate, anomaly score, pressure — all updating live |
| 🚨 Alert Feed | Every detected event logged with timestamp + severity |
| 🔁 Auto Leak Bursts | Automatic periodic leak simulation every ~40 readings |

---

## 🏗 Architecture

```
Ultrasonic Flow Meter
        │
        ▼
┌───────────────────┐
│  Sensor Data      │  flow_rate_lpm · pressure_bar · velocity_ms
│  (CSV / Stream)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  StandardScaler   │  Feature normalization
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Isolation Forest │  Unsupervised anomaly detection
│  (200 estimators) │  contamination = 10%
└────────┬──────────┘
         │
    ┌────┴────┐
    │         │
 NORMAL     ANOMALY
    │         │
    ▼         ▼
 ✅ OK    Severity Classifier
          CRITICAL / HIGH / MEDIUM
               │
               ▼
     ┌─────────────────┐
     │ Streamlit        │
     │ Live Dashboard   │
     └─────────────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| ML Model | Scikit-learn — Isolation Forest |
| Preprocessing | StandardScaler |
| Dashboard | Streamlit |
| Charts | Plotly (Go + Express) |
| Data | Pandas, NumPy |
| Model Persistence | Pickle |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/muhammadsohail613/water-leak-detection.git
cd water-leak-detection

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install pandas numpy scikit-learn streamlit plotly

# 4. Generate synthetic sensor data
python generate_data.py

# 5. Train the AI model
python train_model.py

# 6. Launch the dashboard
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🚀

---

## 📁 Project Structure

```
water-leak-detection/
│
├── app.py                  # Main Streamlit dashboard (Historical + Live tabs)
├── generate_data.py        # Synthetic ultrasonic flow meter data generator
├── train_model.py          # Isolation Forest model training + evaluation
│
├── data/
│   ├── sensor_data.csv     # Raw generated sensor readings
│   └── predictions.csv     # Data enriched with AI predictions + severity
│
├── model/
│   ├── isolation_forest.pkl  # Trained anomaly detection model
│   └── scaler.pkl            # Fitted StandardScaler
│
└── README.md
```

---

## 📈 Model Performance

Trained on 1,000 sensor readings with 3 injected leak events (100 anomalous points):

```
──────────────────────────────────────────────────
              precision    recall  f1-score
──────────────────────────────────────────────────
  Normal         1.00       1.00     1.00
  Leak           0.99       0.99     0.99
──────────────────────────────────────────────────
  Accuracy                           1.00
──────────────────────────────────────────────────

Confusion Matrix:
  ✅ True Negatives  (Normal correctly identified) : 899
  ✅ True Positives  (Leaks correctly caught)      :  99
  ⚠️  False Positives (False alarms)                :   1
  ❌ False Negatives (Missed leaks)                 :   1
──────────────────────────────────────────────────
```

> **99% accuracy · 99% precision · Only 1 false alarm per 1,000 readings**

---

## 🏭 Use Cases

- 🏗 **Industrial pipelines** — continuous monitoring of large-scale pipe networks
- 🏢 **Smart buildings** — integration with BMS for automated leak response
- 🌊 **Water utilities** — municipal water distribution loss detection
- 🏭 **Manufacturing plants** — process water monitoring and loss prevention
- 🔬 **Research** — baseline for training on real ultrasonic sensor hardware

---

## 👨‍💻 Author

<div align="center">

**Muhammad Sohail**

[![Upwork](https://img.shields.io/badge/Upwork-Top%20Rated-6fda44?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com)
[![GitHub](https://img.shields.io/badge/GitHub-muhammadsohail613-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/muhammadsohail613)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077b5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammadsohail613)

BS Artificial Intelligence · 50+ Projects · 100% Job Success Score on Upwork

</div>

---

<div align="center">

⭐ **Star this repo if you found it useful!**

*Built as a freelance portfolio demonstration for AI-powered IoT monitoring systems*

</div>
