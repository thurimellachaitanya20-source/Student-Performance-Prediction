# 🎓 EduPredict AI — Student Performance Prediction & Analytics

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Project%20Status-Complete%20%26%20Production%20Ready-brightgreen.svg)]()

> An end-to-end Machine Learning intelligence platform and interactive academic performance forecaster. Features a **Modern Streamlit Web App**, a **Native Desktop GUI**, multiple benchmarked regression models ($R^2 \approx 87.3\%$), batch student evaluation, reverse goal-seeking, and interactive EDA dashboards.

---

## 🌟 Key Highlights & Features

### 1. 🔮 Interactive Web Application (`app.py`)
- **Smart Forecaster:** Sliders with real-time score updates, half-donut gauge meters, grade badges (Distinction, Merit, Pass, At-Risk), and tailored AI study advice.
- **Quick Preset Profiles:** One-click presets for *Top Achiever*, *Average Performer*, *At-Risk Student*, and *Dedicated Worker*.
- **📂 Batch Cohort Prediction:** Upload a class roster CSV or use the built-in 50-student sample cohort. View pass rate KPIs, grade distribution histograms, and export predicted grades as CSV.
- **📊 Exploratory Data Analysis (EDA):** Interactive Plotly correlation heatmaps, feature distribution histograms with box plots, and OLS regression trendlines.
- **🧠 Model Benchmark Lab:** Side-by-side comparison of 4 algorithms on $R^2$, RMSE, MAE, and 5-fold cross-validation. Displays the exact learned regression equation and relative feature importance percentages.
- **🎯 Goal-Seek Simulator:** Enter a target final score (e.g., 85%), and calculate the exact daily study hours and attendance needed to reach it.
- **⏱️ Study Hub & Pomodoro Timer:** Built-in 25-minute focus sprint timer and high-yield study methodology guide.

### 2. 💻 Native Desktop GUI Suite (`gui_app.py`)
- Built with Python's native `tkinter` / `ttk` with a modern dark theme.
- **Runs 100% offline** with zero browser or web server dependencies.
- Real-time prediction sliders (the score and grade dynamically update as you drag sliders).
- Includes tabs for *Smart Predictor*, *Goal Seek Simulator*, and *Model Benchmarks*.

### 3. 🧪 Production-Grade Machine Learning Pipeline
- Trained on a realistic, statistically validated dataset of **1,000 student records**.
- Compares **Linear Regression**, **Ridge Regression**, **Random Forest Regressor**, and **Gradient Boosting**.
- **Self-Healing Architecture:** Auto-checks if model artifacts exist; if missing, it automatically trains and saves them with zero crashes.

---

## 📊 Model Performance Leaderboard

| Machine Learning Model | $R^2$ Score (Accuracy) | RMSE (Marks) | MAE (Marks) | 5-Fold Cross-Validation $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **🏆 Linear Regression (Best)** | **0.8727** | **3.25** | **2.66** | **0.8784 ± 0.012** |
| **Ridge Regression** | 0.8727 | 3.25 | 2.66 | 0.8784 ± 0.012 |
| **Gradient Boosting Regressor** | 0.8426 | 3.61 | 2.91 | 0.8546 ± 0.018 |
| **Random Forest Regressor** | 0.8116 | 3.95 | 3.16 | 0.8347 ± 0.021 |

> **Learned Formulation:**  
> $\text{Final Marks} \approx -3.50 + 2.75(\text{Study Hours}) + 0.26(\text{Attendance}) + 0.44(\text{Previous Marks})$

---

## 📁 Project Directory Structure

```text
student-performance-prediction/
├── app.py                         # Modern Streamlit Web Application
├── gui_app.py                     # Native Tkinter Desktop Application
├── run_app.bat                    # One-click Windows Web App launcher
├── run_gui.bat                    # One-click Windows Desktop GUI launcher
├── train_model.bat                # One-click Windows Model Training launcher
├── requirements.txt               # Production Python dependencies
├── VIVA_PRESENTATION_GUIDE.md     # Mini-project viva Q&A & presentation guide
├── data/
│   ├── student_performance.csv    # 1,000-record core dataset
│   └── student_performance_dataset.csv # Full multi-feature academic dataset
├── models/
│   ├── student_performance_model.joblib # Serialized model & metrics artifact
│   ├── metrics.json               # JSON benchmark evaluation metrics
│   └── .gitkeep
├── notebooks/
│   └── student_performance_prediction.ipynb # Complete EDA & ML walkthrough notebook
└── src/
    ├── train.py                   # Multi-model training and evaluation script
    └── generate_dataset.py        # Statistical synthetic dataset generator
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Set Up Virtual Environment (Recommended: Python 3.10 – 3.12)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Applications

#### Option A: Streamlit Interactive Web App (Recommended)
```bash
streamlit run app.py
```
*Or on Windows, simply double-click `run_app.bat`!*  
The app will open automatically in your browser at `http://localhost:8501`.

#### Option B: Standalone Desktop GUI
```bash
python gui_app.py
```
*Or on Windows, simply double-click `run_gui.bat`!*

#### Option C: Retrain ML Models
```bash
python src/train.py
```
*Or on Windows, simply double-click `train_model.bat`!*

#### Option D: Jupyter Walkthrough Notebook
```bash
jupyter notebook notebooks/student_performance_prediction.ipynb
```

---

## 🎓 Academic Mini-Project Presentation & Viva

Are you presenting this for a college project review, viva, or interview?  
Check out our dedicated **[Viva & Presentation Guide](VIVA_PRESENTATION_GUIDE.md)** which contains:
- 💡 Complete 2-minute elevator pitch script.
- ❓ Top 10 viva questions asked by external examiners with model answers.
- 📐 Mathematical formulation of Linear Regression vs Random Forest.
- 📋 Live demonstration checklist.

---

## 🛠️ Technology Stack

- **Core Language:** Python 3.12 / 3.11 / 3.10
- **Data Engineering:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, Joblib
- **Web Application:** Streamlit, Streamlit Components
- **Interactive Visualization:** Plotly Graph Objects & Express
- **Desktop Interface:** Tkinter, ttk
- **Notebook & Research:** Jupyter, Matplotlib, Seaborn

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and present this project for academic and personal learning purposes.
