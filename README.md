# Student Performance Prediction

A beginner-friendly machine-learning mini-application that estimates final marks from study hours, attendance, and previous marks. It includes a Jupyter notebook for learning, a reproducible training script, and a Streamlit interface for interactive predictions.

## Features

- Trains a multiple linear regression model with scikit-learn.
- Saves the trained model with Joblib and loads it in the web interface.
- Includes sample data and a Jupyter notebook to explore the workflow.
- Provides an interactive Streamlit form for predictions.

## Project structure

```text
student-performance-prediction/
├── app.py                         # Streamlit interface
├── data/student_performance.csv   # Small illustrative dataset
├── models/                        # Saved model and evaluation metrics
├── notebooks/                     # Learning notebook
├── src/train.py                   # Model training script
├── requirements.txt
└── README.md
```

## Setup

1. Use Python 3.10–3.14. Clone or download this repository, then open a terminal in the project folder.
2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

3. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Train and save the model:

   ```bash
   python src/train.py
   ```

5. Start the local app:

   ```bash
   streamlit run app.py
   ```

The terminal will show a local address, commonly `http://localhost:8501`. Open it in a browser.

## Notebook

Open `notebooks/student_performance_prediction.ipynb` in VS Code or Jupyter and run its code cell. It calls the same training function used by the app, keeping the learning workflow and application model consistent.

## Limitations and responsible use

- The included dataset contains only 15 illustrative, manually created records; it does not represent a real student population.
- Metrics from this dataset are not evidence of real-world accuracy, fairness, or generalisation.
- Do not use this app for grades, admissions, intervention decisions, or any other high-impact educational decision.
- A stronger project would use a larger, consented dataset, investigate bias, validate on separate data, and include input from education professionals.

## GitHub vs. live hosting

Publishing this repository to GitHub makes the source code viewable and shareable. It does **not** make the Streamlit app live. A separate deployment service (such as Streamlit Community Cloud) is needed before you can share a public web-app link.

## Resume entry

**Student Performance Prediction | Python, Pandas, scikit-learn, Streamlit**

- Built a multiple linear regression learning project that estimates final marks from study hours, attendance, and previous marks.
- Created a reproducible training workflow and saved/loaded the model with Joblib.
- Developed a local Streamlit interface for interactive predictions and documented responsible-use limitations.
