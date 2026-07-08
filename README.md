# Fraud Detection System for E-Commerce

An end-to-end machine learning pipeline built to identify fraudulent transaction patterns in e-commerce data. This system implements automated data cleaning, geolocation mapping, domain-specific feature engineering, class imbalance handling, and model artifact serialization.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ installed. Clone this repository and set up your virtual environment:

```bash
# Clone the repository
git clone <your-repository-url>
cd fraud-detection

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
├── data/
│   ├── raw/                 # Original source data (Fraud_Data.csv, etc.)
│   └── processed/           # Engineered data ready for training
├── models/                  # Serialized trained model files (.pkl)
├── notebooks/               # Interactive exploration and analysis
├── scripts/                 # Production-ready, modular Python scripts
│   ├── data_preprocessing.py
│   └── train_model.py
└── README.md


🛠️ Pipeline Architecture
Phase 1: Preprocessing & Geolocation Mapping
File: scripts/data_preprocessing.py

Details: Converts timestamps into explicitly typed datetime features. Resolves raw integer IP addresses to their respective countries via an optimized interval comparison (pd.merge_asof) against global IP network boundaries, handling unknown scopes defensively.

Phase 2: Feature Engineering
Extracts high-signal behavioral features to expose anomalies:

hour: Identifies high-velocity transactional spikes during off-hours.

day_of_week: Detects weekend vs. weekday purchase anomalies.

time_since_signup: Captures immediate post-registration conversion behavior common in automated bot attacks.

Phase 3: Resampling & Model Training
File: scripts/train_model.py

Details: Isolates validation tracking using a stratified split. Addresses severe class imbalance (approx. 9% fraud) dynamically with SMOTE on the training subset only. Trains a Random Forest baseline and explicitly exports the trained model weights directly to the models/ directory for downstream inference/deployment.
```
