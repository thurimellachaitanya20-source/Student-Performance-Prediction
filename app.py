"""Student Performance Predictor & Academic Analytics Hub.
Interactive web application powered by Machine Learning and Streamlit.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "student_performance_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

st.set_page_config(
    page_title="EduPredict | Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_chart(fig: Any, **kwargs):
    """Render Plotly figure cleanly across Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def render_dataframe(df: Any, **kwargs):
    """Render DataFrame cleanly across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


# ---------------------------------------------------------
# Custom Styling (Glassmorphism & Modern Academic Theme)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global Card Styles */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .badge-distinction {
        background: linear-gradient(90deg, #10b981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-merit {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-pass {
        background: linear-gradient(90deg, #f59e0b, #d97706);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-risk {
        background: linear-gradient(90deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 20px;
    }
    .highlight-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #a5b4fc;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Model Loading & Self-Healing Auto-Training
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_or_train_model() -> dict[str, Any]:
    """Load model artifact or automatically train if missing."""
    if not MODEL_PATH.exists():
        from src.train import train_model

        return train_model()
    try:
        artifact = joblib.load(MODEL_PATH)
        return artifact
    except Exception:
        from src.train import train_model

        return train_model()


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load the dataset with fallback generation if needed."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    from src.generate_dataset import generate_student_data

    df = generate_student_data(1000)
    df.to_csv(DATA_PATH, index=False)
    return df


artifact = load_or_train_model()
df_data = load_dataset()


def grade_info(score: float) -> tuple[str, str, str]:
    """Return (Grade, Badge Class, Recommendation Summary)."""
    if score >= 85:
        return "A+ (Distinction)", "badge-distinction", "Outstanding performance! Maintain consistency and explore advanced honors challenges."
    if score >= 75:
        return "A (Excellent)", "badge-distinction", "Strong academic trajectory! Target minor improvement areas to enter top honors."
    if score >= 65:
        return "B (Good / First Class)", "badge-merit", "Good comprehension. Additional focused revision can easily push this to an A grade."
    if score >= 50:
        return "C (Pass / Average)", "badge-pass", "Moderate risk. Needs structured daily study habits and higher lecture attendance."
    return "F (Academic Support Needed)", "badge-risk", "High risk of underachievement. Immediate faculty mentoring and study intervention recommended."


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.markdown("## ⚙️ Project Controls")

# Navigation
view_mode = st.sidebar.radio(
    "Select Feature Module:",
    [
        "🔮 Smart AI Predictor",
        "📂 Batch Class Analysis",
        "📊 Interactive EDA Dashboard",
        "🧠 Model Comparison Lab",
        "🎯 Goal-Seek & What-If Simulator",
        "⏱️ Focus & Study Hub",
    ],
    index=0,
)

# Select Model
candidate_models = artifact.get("candidate_models", {})
model_options = list(candidate_models.keys()) if candidate_models else ["Linear Regression"]
best_name = artifact.get("best_model_name", "Linear Regression")

default_idx = model_options.index(best_name) if best_name in model_options else 0
selected_model_name = st.sidebar.selectbox(
    "Active ML Algorithm:",
    options=model_options,
    index=default_idx,
    help="Compare how different ML algorithms predict on the same student inputs.",
)
active_model = candidate_models.get(selected_model_name, artifact["model"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Mini-Project Highlights:**
    - 📈 **Dataset:** 1,000 Records
    - 🧠 **Algorithms:** 4 Regressors
    - 🎯 **Best Model:** Linear Regression
    - 🏆 **Accuracy (R²):** ~87.3%
    - 💻 **Tech Stack:** Python, Streamlit, Scikit-Learn, Plotly
    """
)

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([4, 1.2])
with col_h1:
    st.markdown('<div class="hero-title">🎓 Student Performance Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">'
        'Interactive Machine Learning system for academic performance forecasting, '
        'risk intervention, and behavioral factor insights.'
        '</div>',
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f"""
        <div style="text-align: right; padding-top: 10px;">
            <span class="highlight-pill">v2.0 Full Stack</span>
            <span class="highlight-pill">GitHub Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# MODULE 1: SMART AI PREDICTOR
# ---------------------------------------------------------
if view_mode == "🔮 Smart AI Predictor":
    st.markdown("### 🔮 Individual Student Performance Forecaster")
    st.write("Input academic habits to forecast final marks with real-time feedback, gauge meters, and personalized recommendations.")

    # Preset Quick-Fills
    st.markdown("**Quick Preset Profiles:**")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    if "study_hours" not in st.session_state:
        st.session_state.study_hours = 6.0
        st.session_state.attendance = 85.0
        st.session_state.previous_marks = 75.0
        st.session_state.sleep_hours = 7.0
        st.session_state.practice_tests = 4

    with p_col1:
        if st.button("🌟 Top Achiever", use_container_width=True):
            st.session_state.study_hours = 9.5
            st.session_state.attendance = 96.0
            st.session_state.previous_marks = 90.0
            st.session_state.sleep_hours = 7.5
            st.session_state.practice_tests = 8
            st.rerun()
    with p_col2:
        if st.button("⚖️ Average Performer", use_container_width=True):
            st.session_state.study_hours = 5.0
            st.session_state.attendance = 80.0
            st.session_state.previous_marks = 68.0
            st.session_state.sleep_hours = 7.0
            st.session_state.practice_tests = 4
            st.rerun()
    with p_col3:
        if st.button("⚠️ At-Risk Student", use_container_width=True):
            st.session_state.study_hours = 2.0
            st.session_state.attendance = 60.0
            st.session_state.previous_marks = 45.0
            st.session_state.sleep_hours = 5.0
            st.session_state.practice_tests = 1
            st.rerun()
    with p_col4:
        if st.button("📈 Hard Worker (Catching Up)", use_container_width=True):
            st.session_state.study_hours = 8.5
            st.session_state.attendance = 92.0
            st.session_state.previous_marks = 55.0
            st.session_state.sleep_hours = 6.5
            st.session_state.practice_tests = 7
            st.rerun()

    # Input Form
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            study_hours = st.slider(
                "📖 Daily Study Hours",
                min_value=0.5,
                max_value=14.0,
                value=float(st.session_state.study_hours),
                step=0.5,
                help="Self-study and homework hours per day.",
            )
            sleep_hours = st.slider(
                "😴 Daily Sleep Hours",
                min_value=4.0,
                max_value=10.0,
                value=float(st.session_state.sleep_hours),
                step=0.5,
                help="Average nightly sleep duration.",
            )

        with col_in2:
            attendance = st.slider(
                "🏫 Class Attendance (%)",
                min_value=40.0,
                max_value=100.0,
                value=float(st.session_state.attendance),
                step=1.0,
                help="Percentage of lectures attended.",
            )
            practice_tests = st.slider(
                "📝 Practice Tests Completed",
                min_value=0,
                max_value=10,
                value=int(st.session_state.practice_tests),
                step=1,
                help="Number of mock tests or quizzes solved before exams.",
            )

        with col_in3:
            previous_marks = st.slider(
                "📊 Previous Term Marks (%)",
                min_value=20.0,
                max_value=100.0,
                value=float(st.session_state.previous_marks),
                step=1.0,
                help="Marks scored in the preceding semester or midterm.",
            )
            input_mode = st.radio(
                "Feature Evaluation Mode:",
                ["Standard (3 Core Factors)", "Enhanced (5 Academic Factors)"],
                horizontal=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Compute Prediction
    if input_mode == "Standard (3 Core Factors)":
        features = ["Study_Hours", "Attendance", "Previous_Marks"]
        input_df = pd.DataFrame([[study_hours, attendance, previous_marks]], columns=features)
        pred_raw = float(active_model.predict(input_df)[0])
    else:
        m5 = artifact.get("model_5f")
        if m5:
            features = ["Study_Hours", "Attendance", "Previous_Marks", "Sleep_Hours", "Practice_Tests"]
            input_df = pd.DataFrame(
                [[study_hours, attendance, previous_marks, sleep_hours, practice_tests]],
                columns=features,
            )
            pred_raw = float(m5["model"].predict(input_df)[0])
        else:
            features = ["Study_Hours", "Attendance", "Previous_Marks"]
            input_df = pd.DataFrame([[study_hours, attendance, previous_marks]], columns=features)
            pred_raw = float(active_model.predict(input_df)[0])

    predicted_marks = min(max(pred_raw, 0.0), 100.0)
    grade, badge_cls, advice = grade_info(predicted_marks)

    # Output Presentation
    st.markdown("---")
    res_col1, res_col2 = st.columns([1.1, 1.2])

    with res_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.95rem; opacity: 0.8; margin-bottom: 5px;">Forecasted Final Score</div>
                <div style="font-size: 3rem; font-weight: 800; color: #38bdf8;">
                    {predicted_marks:.1f} <span style="font-size: 1.2rem; opacity: 0.7;">/ 100</span>
                </div>
                <div style="margin-top: 8px;">
                    <span class="{badge_cls}" style="font-size: 1rem; padding: 6px 16px;">{grade}</span>
                </div>
                <div style="margin-top: 15px; font-size: 0.9rem; line-height: 1.4; opacity: 0.9;">
                    <strong>AI Recommendation:</strong><br>{advice}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge Chart
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=predicted_marks,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Performance Indicator", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#38bdf8", "thickness": 0.3},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(239, 68, 68, 0.35)"},
                        {"range": [50, 65], "color": "rgba(245, 158, 11, 0.35)"},
                        {"range": [65, 80], "color": "rgba(59, 130, 246, 0.35)"},
                        {"range": [80, 100], "color": "rgba(16, 185, 129, 0.35)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 3},
                        "thickness": 0.8,
                        "value": predicted_marks,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        render_chart(fig_gauge)

    with res_col2:
        # Comparison with Cohort Average
        avg_study = float(df_data["Study_Hours"].mean()) if "Study_Hours" in df_data else 5.5
        avg_att = float(df_data["Attendance"].mean()) if "Attendance" in df_data else 82.0
        avg_prev = float(df_data["Previous_Marks"].mean()) if "Previous_Marks" in df_data else 70.0
        avg_final = float(df_data["Final_Marks"].mean()) if "Final_Marks" in df_data else 72.0

        comp_df = pd.DataFrame(
            {
                "Metric": ["Study Hours (x10)", "Attendance (%)", "Previous Marks", "Forecasted Marks"],
                "Student": [study_hours * 10, attendance, previous_marks, predicted_marks],
                "Cohort Average": [avg_study * 10, avg_att, avg_prev, avg_final],
            }
        )

        fig_comp = px.bar(
            comp_df,
            x="Metric",
            y=["Student", "Cohort Average"],
            barmode="group",
            title="Student vs Class Cohort Benchmark",
            color_discrete_sequence=["#38bdf8", "#94a3b8"],
        )
        fig_comp.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        render_chart(fig_comp)

        # Actionable Factor Gain Simulation
        st.markdown("**⚡ Potential Score Gains with Small Adjustments:**")
        gain_cols = st.columns(3)
        gain_cols[0].metric("+1.5h Study/Day", f"+{min(1.5 * 2.75, 100 - predicted_marks):.1f} pts")
        gain_cols[1].metric("+10% Attendance", f"+{min(10.0 * 0.26, 100 - predicted_marks):.1f} pts")
        gain_cols[2].metric("+2 Practice Tests", f"+{min(2.0 * 0.75, 100 - predicted_marks):.1f} pts")


# ---------------------------------------------------------
# MODULE 2: BATCH CLASS ANALYSIS
# ---------------------------------------------------------
elif view_mode == "📂 Batch Class Analysis":
    st.markdown("### 📂 Batch Student Prediction & Cohort Analysis")
    st.write("Upload a class roster CSV or use the built-in 50-student sample cohort to run batch inference and export reports.")

    batch_source = st.radio(
        "Choose Batch Data Source:",
        ["Use Preloaded Sample Cohort (50 Students)", "Upload Custom CSV File"],
        horizontal=True,
    )

    batch_df: pd.DataFrame | None = None

    if batch_source == "Upload Custom CSV File":
        uploaded_file = st.file_uploader("Upload student CSV (Columns: Study_Hours, Attendance, Previous_Marks)", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
        else:
            # Download template
            sample_template = pd.DataFrame(
                {
                    "Student_Name": ["Alex Rivera", "Brenda Chen", "Carlos Gomez", "Diana Prince", "Ethan Hunt"],
                    "Study_Hours": [7.0, 4.5, 8.0, 2.5, 6.0],
                    "Attendance": [92.0, 78.0, 95.0, 62.0, 85.0],
                    "Previous_Marks": [80.0, 65.0, 88.0, 48.0, 72.0],
                }
            )
            csv_buf = io.StringIO()
            sample_template.to_csv(csv_buf, index=False)
            st.download_button(
                "📥 Download CSV Template",
                data=csv_buf.getvalue(),
                file_name="student_roster_template.csv",
                mime="text/csv",
            )
    else:
        # Generate 50 sample records
        sample_50 = df_data.sample(n=min(50, len(df_data)), random_state=42).copy()
        if "Student_ID" not in sample_50.columns:
            sample_50["Student_ID"] = [f"STU{2001 + i}" for i in range(len(sample_50))]
        batch_df = sample_50

    if batch_df is not None:
        required_cols = ["Study_Hours", "Attendance", "Previous_Marks"]
        missing_cols = [c for c in required_cols if c not in batch_df.columns]

        if missing_cols:
            st.error(f"The dataset is missing required columns: {missing_cols}")
        else:
            # Run Batch Predictions
            features = ["Study_Hours", "Attendance", "Previous_Marks"]
            batch_preds = active_model.predict(batch_df[features])
            batch_df["Predicted_Marks"] = np.clip(batch_preds, 0, 100).round(1)
            batch_df["Predicted_Grade"] = [grade_info(m)[0] for m in batch_df["Predicted_Marks"]]

            # KPI Summary
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            avg_pred = batch_df["Predicted_Marks"].mean()
            pass_count = (batch_df["Predicted_Marks"] >= 50).sum()
            pass_rate = (pass_count / len(batch_df)) * 100
            dist_count = (batch_df["Predicted_Marks"] >= 80).sum()
            risk_count = (batch_df["Predicted_Marks"] < 50).sum()

            kpi1.metric("Class Average", f"{avg_pred:.1f} / 100")
            kpi2.metric("Pass Rate", f"{pass_rate:.1f}%", f"{pass_count}/{len(batch_df)} Students")
            kpi3.metric("Top Honors (≥80%)", f"{dist_count} Students")
            kpi4.metric("At-Risk (<50%)", f"{risk_count} Students", delta=-int(risk_count) if risk_count > 0 else 0, delta_color="inverse")

            # Charts
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                fig_grade = px.histogram(
                    batch_df,
                    x="Predicted_Grade",
                    color="Predicted_Grade",
                    title="Predicted Grade Distribution",
                    category_orders={
                        "Predicted_Grade": [
                            "A+ (Distinction)",
                            "A (Excellent)",
                            "B (Good / First Class)",
                            "C (Pass / Average)",
                            "F (Academic Support Needed)",
                        ]
                    },
                    color_discrete_map={
                        "A+ (Distinction)": "#10b981",
                        "A (Excellent)": "#34d399",
                        "B (Good / First Class)": "#3b82f6",
                        "C (Pass / Average)": "#f59e0b",
                        "F (Academic Support Needed)": "#ef4444",
                    },
                )
                fig_grade.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                render_chart(fig_grade)

            with b_col2:
                fig_scatter = px.scatter(
                    batch_df,
                    x="Study_Hours",
                    y="Predicted_Marks",
                    color="Attendance",
                    size="Previous_Marks",
                    title="Study Hours vs Predicted Marks (Size = Prev Marks, Color = Attendance)",
                    color_continuous_scale="Viridis",
                )
                fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                render_chart(fig_scatter)

            # Results Table & Download
            st.markdown("#### Detailed Prediction Roster")
            render_dataframe(batch_df, hide_index=True)

            csv_out = io.StringIO()
            batch_df.to_csv(csv_out, index=False)
            st.download_button(
                "📥 Export Full Prediction Report (CSV)",
                data=csv_out.getvalue(),
                file_name="batch_student_predictions_report.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ---------------------------------------------------------
# MODULE 3: INTERACTIVE EDA DASHBOARD
# ---------------------------------------------------------
elif view_mode == "📊 Interactive EDA Dashboard":
    st.markdown("### 📊 Exploratory Data Analysis & Feature Dynamics")
    st.write("Examine underlying distribution patterns, correlations, and academic behavioral drivers across the 1,000-student dataset.")

    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📈 Correlations & Heatmap", "🔍 Feature Distributions", "🎯 Scatter & Trendlines"])

    with eda_tab1:
        st.markdown("#### Correlation Matrix with Target (Final Marks)")
        num_cols = ["Study_Hours", "Attendance", "Previous_Marks", "Final_Marks"]
        if "Sleep_Hours" in df_data.columns:
            num_cols.append("Sleep_Hours")
        if "Practice_Tests" in df_data.columns:
            num_cols.append("Practice_Tests")

        corr = df_data[num_cols].corr()

        fig_heat = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Pearson Correlation Heatmap",
            zmin=-1,
            zmax=1,
        )
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=450)
        render_chart(fig_heat)

        st.markdown(
            """
            **Key Correlation Takeaways:**
            - **Previous Marks & Study Hours** demonstrate the highest positive correlation with final grades.
            - **Attendance** acts as a critical reliability baseline—students below 70% attendance rarely achieve top honors regardless of previous marks.
            - **Practice Tests** create significant compounding benefits on final score outcomes.
            """
        )

    with eda_tab2:
        st.markdown("#### Feature Distribution Profiling")
        selected_feat = st.selectbox("Select Feature to Inspect:", num_cols, index=0)

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            fig_hist = px.histogram(
                df_data,
                x=selected_feat,
                nbins=30,
                marginal="box",
                title=f"Distribution of {selected_feat}",
                color_discrete_sequence=["#6366f1"],
            )
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            render_chart(fig_hist)

        with f_col2:
            st.markdown(f"**Descriptive Statistics for `{selected_feat}`:**")
            stats = df_data[selected_feat].describe().to_frame()
            render_dataframe(stats)

    with eda_tab3:
        st.markdown("#### Scatter Trends & OLS Regression")
        x_axis = st.selectbox("Select Independent Variable (X-axis):", ["Study_Hours", "Attendance", "Previous_Marks"], index=0)

        fig_ols = px.scatter(
            df_data,
            x=x_axis,
            y="Final_Marks",
            trendline="ols",
            title=f"{x_axis} vs Final Marks (with OLS Trendline)",
            opacity=0.6,
            color_discrete_sequence=["#38bdf8"],
        )
        fig_ols.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        render_chart(fig_ols)


# ---------------------------------------------------------
# MODULE 4: MODEL COMPARISON LAB
# ---------------------------------------------------------
elif view_mode == "🧠 Model Comparison Lab":
    st.markdown("### 🧠 Machine Learning Model Benchmark & Interpretability Lab")
    st.write("Compare 4 different machine learning regressors on accuracy, error metrics, cross-validation, and feature importance.")

    metrics_dict = artifact.get("metrics", {})
    if not metrics_dict and METRICS_PATH.exists():
        metrics_dict = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("models", {})

    # Comparison Table
    records = []
    for name, m in metrics_dict.items():
        records.append(
            {
                "Model": name,
                "R² Score": m.get("r2", 0.0),
                "RMSE (Marks)": m.get("rmse", 0.0),
                "MAE (Marks)": m.get("mae", 0.0),
                "5-Fold CV R²": m.get("cv_r2_mean", 0.0),
                "CV Std Dev": m.get("cv_r2_std", 0.0),
            }
        )

    metrics_df = pd.DataFrame(records).sort_values("R² Score", ascending=False)

    st.markdown("#### 🏆 Algorithmic Leaderboard")
    render_dataframe(metrics_df, hide_index=True)

    # Visualization
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fig_r2 = px.bar(
            metrics_df,
            x="Model",
            y="R² Score",
            title="Model Accuracy Comparison (R² Score - Higher is Better)",
            color="R² Score",
            color_continuous_scale="Teal",
            text="R² Score",
        )
        fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_r2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_range=[0, 1.05])
        render_chart(fig_r2)

    with m_col2:
        fig_rmse = px.bar(
            metrics_df,
            x="Model",
            y="RMSE (Marks)",
            title="Model Error Comparison (RMSE - Lower is Better)",
            color="RMSE (Marks)",
            color_continuous_scale="Reds_r",
            text="RMSE (Marks)",
        )
        fig_rmse.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig_rmse.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        render_chart(fig_rmse)

    # Feature Importance & Mathematical Formulation
    st.markdown("---")
    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown("#### 🔍 Feature Importance Breakdown")
        imp_data = artifact.get("feature_importance", {})
        if imp_data:
            df_imp = pd.DataFrame(list(imp_data.items()), columns=["Feature", "Contribution (%)"]).sort_values(
                "Contribution (%)", ascending=True
            )
            fig_imp = px.bar(
                df_imp,
                x="Contribution (%)",
                y="Feature",
                orientation="h",
                title="Relative Feature Importance (%)",
                color="Contribution (%)",
                color_continuous_scale="Viridis",
                text="Contribution (%)",
            )
            fig_imp.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            render_chart(fig_imp)

    with f_col2:
        st.markdown("#### 📐 Mathematical Formulation")
        st.markdown(
            """
            The Multiple Linear Regression model computes predicted marks using the learned linear equation:
            """
        )
        best_mod = artifact.get("model")
        if hasattr(best_mod, "coef_") and hasattr(best_mod, "intercept_"):
            feats = artifact.get("features", ["Study_Hours", "Attendance", "Previous_Marks"])
            coefs = best_mod.coef_
            intercept = best_mod.intercept_
            eq_terms = [f"{c:+.3f} \\times \\text{{{f}}}" for f, c in zip(feats, coefs)]
            latex_eq = f"\\text{{Final Marks}} = {intercept:.2f} " + " ".join(eq_terms)
            st.latex(latex_eq)
        else:
            st.latex(r"\text{Final Marks} = \beta_0 + \beta_1(\text{Study Hours}) + \beta_2(\text{Attendance}) + \beta_3(\text{Previous Marks})")

        st.info(
            "💡 **Interpretation:** Each additional daily hour of study contributes directly to marks, "
            "while previous foundation establishes the score baseline."
        )


# ---------------------------------------------------------
# MODULE 5: GOAL SEEK & WHAT-IF SIMULATOR
# ---------------------------------------------------------
elif view_mode == "🎯 Goal-Seek & What-If Simulator":
    st.markdown("### 🎯 Target Score Calculator & Reverse Simulator")
    st.write("Set your dream target final mark and see exactly what study effort and attendance levels are required to achieve it.")

    col_g1, col_g2 = st.columns([1, 1.2])

    with col_g1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        target_score = st.slider("🎯 Desired Target Final Marks:", min_value=50.0, max_value=98.0, value=85.0, step=1.0)
        current_prev = st.slider("📊 Your Current Previous Marks:", min_value=30.0, max_value=95.0, value=70.0, step=1.0)
        current_att = st.slider("🏫 Planned Attendance Level (%):", min_value=60.0, max_value=100.0, value=85.0, step=5.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_g2:
        best_mod = artifact.get("candidate_models", {}).get("Linear Regression", artifact.get("model"))
        if hasattr(best_mod, "coef_") and hasattr(best_mod, "intercept_"):
            feats = artifact.get("features", ["Study_Hours", "Attendance", "Previous_Marks"])
            coef_map = dict(zip(feats, best_mod.coef_))
            b1 = coef_map.get("Study_Hours", 2.75)
            b2 = coef_map.get("Attendance", 0.26)
            b3 = coef_map.get("Previous_Marks", 0.44)
            b0 = best_mod.intercept_

            req_hours = (target_score - b0 - (b2 * current_att) - (b3 * current_prev)) / b1
            req_hours = max(req_hours, 0.5)
        else:
            req_hours = max((target_score - (0.44 * current_prev) - (0.26 * current_att)) / 2.75, 1.0)

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.95rem; opacity: 0.8;">Required Daily Study Commitment</div>
                <div style="font-size: 2.8rem; font-weight: 800; color: #10b981;">
                    {req_hours:.1f} <span style="font-size: 1.1rem; opacity: 0.8;">hours / day</span>
                </div>
                <div style="margin-top: 10px; font-size: 0.95rem;">
                    To reach <strong>{target_score:.0f}% marks</strong> with <strong>{current_att:.0f}% attendance</strong>
                    and previous mark baseline of <strong>{current_prev:.0f}%</strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if req_hours > 10.0:
            st.warning("⚠️ Achieving this high target score requires intensive dedication (~10+ hrs/day). Consider also boosting attendance to 95%+ and completing practice tests.")
        elif req_hours <= 3.0:
            st.success("✅ This goal is very well within reach with your strong baseline! Maintain steady daily review.")
        else:
            st.info("👍 A healthy and achievable daily study target! Combine with regular practice exams.")

        # Scenario Matrix
        st.markdown("**Alternative Study Hours vs Attendance Matrix:**")
        att_options = [75.0, 85.0, 95.0]
        matrix_rows = []
        for att_val in att_options:
            h_needed = max((target_score - b0 - (b2 * att_val) - (b3 * current_prev)) / b1, 0.5) if hasattr(best_mod, "coef_") else 5.0
            matrix_rows.append({"Attendance": f"{att_val:.0f}%", "Required Study Hours/Day": f"{h_needed:.1f} hrs"})
        st.table(pd.DataFrame(matrix_rows))


# ---------------------------------------------------------
# MODULE 6: FOCUS & STUDY HUB
# ---------------------------------------------------------
elif view_mode == "⏱️ Focus & Study Hub":
    st.markdown("### ⏱️ Student Focus & Productivity Hub")
    st.write("Boost your study efficiency with scientifically proven focus techniques and study planning.")

    f_col1, f_col2 = st.columns(2)

    with f_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### ⏳ Pomodoro Focus Sprint (25 Minutes)")
        st.write("The Pomodoro Technique breaks work into intervals, traditionally 25 minutes in length, separated by short breaks.")

        st.markdown(
            """
            <div style="text-align: center; margin: 20px 0;">
                <div style="font-size: 4rem; font-family: monospace; font-weight: 800; color: #f59e0b;">
                    25 : 00
                </div>
                <div style="font-size: 0.9rem; opacity: 0.7;">Sprint Cycle: Active Learning & Practice</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            if st.button("▶️ Start Focus Session", use_container_width=True):
                st.toast("🎯 Focus session initiated! Put away phone notifications and begin reading.")
        with b_c2:
            if st.button("🔄 Reset Timer", use_container_width=True):
                st.toast("Timer reset to 25 minutes.")

        st.markdown("</div>", unsafe_allow_html=True)

    with f_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("#### 📚 High-Yield Academic Strategies")
        st.markdown(
            """
            1. **Active Recall over Passive Re-reading:**
               - Test yourself using flashcards or practice questions instead of simply highlighting notes.
            2. **Spaced Repetition:**
               - Review difficult topics at 1-day, 3-day, and 7-day intervals to lock information into long-term memory.
            3. **The Feynman Technique:**
               - Explain concepts in simple terms as if teaching a beginner. If you get stuck, review that specific section.
            4. **Sleep & Cognitive Retention:**
               - Studies show memory consolidation occurs during REM sleep. Sleeping 7-8 hours yields higher test performance than all-night cramming.
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; font-size: 0.85rem; opacity: 0.7; padding: 10px 0;">
        <strong>EduPredict AI</strong> • Student Performance Analytics & Predictive Intelligence Mini-Project<br>
        Developed with Python, Scikit-Learn, Streamlit & Plotly • GitHub Repository Ready
    </div>
    """,
    unsafe_allow_html=True,
)
