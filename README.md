# Rainfall-Prediction-in-India
# 🌧️ Rainfall Prediction in India

This project predicts rainfall across different regions in India using machine learning models. It includes both classification (Rain: Yes/No) and regression (Rainfall Amount in mm) approaches based on historical rainfall data.

---

## 📌 Project Features

- 📊 **Exploratory Data Analysis (EDA)** on 1901–2015 rainfall data
- 🧠 **ML Models**:
  - Classification: Random Forest, XGBoost
  - Regression: Random Forest, XGBoost
- 🛠️ **Feature Engineering**: Monthly and seasonal aggregations
- 📈 **Model Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- 🌐 **Streamlit App** for real-time rainfall prediction and visualization

---

## 📁 Dataset

- `rainfall in india 1901-2015.csv`: Historical rainfall by region and month
- `district wise rainfall normal.csv`: Long-term district-level rainfall normals

Datasets sourced from [data.gov.in](https://data.gov.in/) and [Kaggle](https://www.kaggle.com/).

---

## 🧰 Installation

### ✅ Requirements

- Python 3.8+
- Streamlit
- Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Joblib

### 🔧 Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/rainfall-prediction.git
cd rainfall-prediction

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
