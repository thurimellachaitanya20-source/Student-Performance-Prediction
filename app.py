from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")
st.title("🎓 Student Performance Predictor")
st.write("Enter a student's details to estimate final marks using a linear regression model.")
st.caption("Educational demo only — this is not a real academic decision tool.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python src/train.py` first.")
    return joblib.load(MODEL_PATH)


try:
    model_artifact = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.form("prediction_form"):
    study_hours = st.slider("Study hours per day", min_value=0.0, max_value=16.0, value=7.0, step=0.5)
    attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=90)
    previous_marks = st.slider("Previous marks", min_value=0, max_value=100, value=75)
    submitted = st.form_submit_button("Predict final marks")

if submitted:
    student = pd.DataFrame(
        [[study_hours, attendance, previous_marks]],
        columns=model_artifact["features"],
    )
    prediction = float(model_artifact["model"].predict(student)[0])
    bounded_prediction = min(max(prediction, 0), 100)
    st.success(f"Estimated final marks: {bounded_prediction:.2f} / 100")
    st.dataframe(student, hide_index=True, use_container_width=True)

with st.expander("About this project"):
    st.write(
        "The model is trained on a small, illustrative dataset with study hours, attendance, "
        "previous marks, and final marks. Its results should not be used for grading or admissions."
    )
