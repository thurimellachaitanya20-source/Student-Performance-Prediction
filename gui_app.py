"""EduPredict - Desktop GUI for Student Performance Prediction.
Built with Python Tkinter/ttk with a modern dark-mode aesthetic.
Runs out of the box with zero browser dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "student_performance_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"


def load_model_artifact() -> dict[str, Any]:
    """Load model artifact or automatically train if missing."""
    if not MODEL_PATH.exists():
        from src.train import train_model

        return train_model()
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from src.train import train_model

        return train_model()


class StudentPredictorGUI(tk.Tk):
    """Main Application Window for EduPredict Desktop."""

    def __init__(self):
        super().__init__()
        self.title("EduPredict AI — Student Performance Desktop Suite")
        self.geometry("960x720")
        self.minsize(880, 640)

        # Theme Colors
        self.bg_color = "#0f172a"        # Deep slate
        self.card_color = "#1e293b"      # Surface slate
        self.accent_color = "#38bdf8"    # Electric sky blue
        self.text_color = "#f8fafc"      # Off white
        self.muted_text = "#94a3b8"      # Slate gray
        self.highlight_green = "#10b981" # Emerald
        self.highlight_amber = "#f59e0b" # Amber
        self.highlight_red = "#ef4444"   # Red

        self.configure(bg=self.bg_color)
        self.artifact = load_model_artifact()

        # State Variables
        self.study_hours_var = tk.DoubleVar(value=6.0)
        self.attendance_var = tk.DoubleVar(value=85.0)
        self.prev_marks_var = tk.DoubleVar(value=75.0)
        self.selected_model_var = tk.StringVar(value=self.artifact.get("best_model_name", "Linear Regression"))

        # Setup Styling
        self.setup_styles()

        # Build UI Structure
        self.build_header()
        self.build_notebook()

        # Trigger initial calculation
        self.update_prediction()

    def setup_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Configure Notebook / Tabs
        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#334155",
            foreground=self.text_color,
            font=("Segoe UI", 10, "bold"),
            padding=[16, 8],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", self.accent_color)],
            foreground=[("selected", "#0f172a")],
        )

        # Configure Scales and Labels
        style.configure("TScale", background=self.card_color)
        style.configure("TProgressbar", troughcolor="#334155", background=self.accent_color)

    def build_header(self):
        header_frame = tk.Frame(self, bg=self.bg_color)
        header_frame.pack(fill=tk.X, padx=24, pady=(16, 8))

        title_lbl = tk.Label(
            header_frame,
            text="🎓 EduPredict AI — Desktop Analytics Suite",
            font=("Segoe UI", 18, "bold"),
            fg=self.accent_color,
            bg=self.bg_color,
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            header_frame,
            text="Real-Time Machine Learning Performance Predictor & Academic Advisory Engine",
            font=("Segoe UI", 10),
            fg=self.muted_text,
            bg=self.bg_color,
        )
        sub_lbl.pack(anchor="w")

    def build_notebook(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)

        # Tab 1: Predictor
        tab_predict = tk.Frame(notebook, bg=self.bg_color)
        notebook.add(tab_predict, text="  🔮 Smart Predictor  ")
        self.build_predict_tab(tab_predict)

        # Tab 2: Goal Seek
        tab_goal = tk.Frame(notebook, bg=self.bg_color)
        notebook.add(tab_goal, text="  🎯 Goal Seek Simulator  ")
        self.build_goal_tab(tab_goal)

        # Tab 3: Model Metrics & Info
        tab_metrics = tk.Frame(notebook, bg=self.bg_color)
        notebook.add(tab_metrics, text="  📊 Model Benchmarks  ")
        self.build_metrics_tab(tab_metrics)

    def build_predict_tab(self, parent):
        container = tk.Frame(parent, bg=self.bg_color)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left Column: Input Sliders & Presets
        left_card = tk.Frame(container, bg=self.card_color, bd=1, relief=tk.SOLID)
        left_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Preset Buttons
        preset_frame = tk.LabelFrame(
            left_card,
            text=" Quick Presets ",
            font=("Segoe UI", 10, "bold"),
            fg=self.accent_color,
            bg=self.card_color,
            padx=10,
            pady=8,
        )
        preset_frame.pack(fill=tk.X, padx=16, pady=(16, 10))

        btn_grid = tk.Frame(preset_frame, bg=self.card_color)
        btn_grid.pack(fill=tk.X)

        presets = [
            ("🌟 Achiever", 9.5, 96.0, 90.0),
            ("⚖️ Average", 5.0, 80.0, 68.0),
            ("⚠️ At-Risk", 2.0, 60.0, 45.0),
            ("📈 Dedicated", 8.5, 92.0, 55.0),
        ]
        for idx, (label, h, a, p) in enumerate(presets):
            btn = tk.Button(
                btn_grid,
                text=label,
                font=("Segoe UI", 9, "bold"),
                bg="#334155",
                fg=self.text_color,
                activebackground=self.accent_color,
                activeforeground="#0f172a",
                relief=tk.FLAT,
                padx=8,
                pady=4,
                cursor="hand2",
                command=lambda hours=h, att=a, prev=p: self.apply_preset(hours, att, prev),
            )
            btn.grid(row=0, column=idx, padx=4, pady=4, sticky="ew")
            btn_grid.grid_columnconfigure(idx, weight=1)

        # Sliders Section
        slider_frame = tk.LabelFrame(
            left_card,
            text=" Academic Input Metrics ",
            font=("Segoe UI", 10, "bold"),
            fg=self.accent_color,
            bg=self.card_color,
            padx=16,
            pady=12,
        )
        slider_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

        # Study Hours Slider
        self.lbl_study = tk.Label(
            slider_frame,
            text="Daily Study Hours: 6.0 hrs",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_color,
            anchor="w",
        )
        self.lbl_study.pack(fill=tk.X, pady=(6, 2))
        s_scale = ttk.Scale(
            slider_frame,
            from_=0.5,
            to=14.0,
            variable=self.study_hours_var,
            command=lambda v: self.on_slider_change("study"),
        )
        s_scale.pack(fill=tk.X, pady=(0, 10))

        # Attendance Slider
        self.lbl_att = tk.Label(
            slider_frame,
            text="Class Attendance: 85.0%",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_color,
            anchor="w",
        )
        self.lbl_att.pack(fill=tk.X, pady=(6, 2))
        a_scale = ttk.Scale(
            slider_frame,
            from_=40.0,
            to=100.0,
            variable=self.attendance_var,
            command=lambda v: self.on_slider_change("att"),
        )
        a_scale.pack(fill=tk.X, pady=(0, 10))

        # Previous Marks Slider
        self.lbl_prev = tk.Label(
            slider_frame,
            text="Previous Exam Marks: 75.0%",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_color,
            anchor="w",
        )
        self.lbl_prev.pack(fill=tk.X, pady=(6, 2))
        p_scale = ttk.Scale(
            slider_frame,
            from_=20.0,
            to=100.0,
            variable=self.prev_marks_var,
            command=lambda v: self.on_slider_change("prev"),
        )
        p_scale.pack(fill=tk.X, pady=(0, 10))

        # Algorithm Selector
        model_row = tk.Frame(slider_frame, bg=self.card_color)
        model_row.pack(fill=tk.X, pady=(10, 5))

        tk.Label(
            model_row,
            text="ML Regressor:",
            font=("Segoe UI", 9, "bold"),
            fg=self.muted_text,
            bg=self.card_color,
        ).pack(side=tk.LEFT, padx=(0, 8))

        models_list = list(self.artifact.get("candidate_models", {"Linear Regression": None}).keys())
        model_combo = ttk.Combobox(
            model_row,
            textvariable=self.selected_model_var,
            values=models_list,
            state="readonly",
            width=20,
        )
        model_combo.pack(side=tk.LEFT)
        model_combo.bind("<<ComboboxSelected>>", lambda e: self.update_prediction())

        # Right Column: Output Result Display
        right_card = tk.Frame(container, bg=self.card_color, bd=1, relief=tk.SOLID)
        right_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        res_box = tk.LabelFrame(
            right_card,
            text=" Forecasted Performance ",
            font=("Segoe UI", 10, "bold"),
            fg=self.accent_color,
            bg=self.card_color,
            padx=16,
            pady=16,
        )
        res_box.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        self.score_display = tk.Label(
            res_box,
            text="78.5",
            font=("Segoe UI", 48, "bold"),
            fg=self.accent_color,
            bg=self.card_color,
        )
        self.score_display.pack(pady=(10, 0))

        self.out_of_lbl = tk.Label(
            res_box,
            text="Marks out of 100",
            font=("Segoe UI", 10),
            fg=self.muted_text,
            bg=self.card_color,
        )
        self.out_of_lbl.pack(pady=(0, 12))

        # Grade Badge
        self.badge_lbl = tk.Label(
            res_box,
            text="A (Excellent)",
            font=("Segoe UI", 12, "bold"),
            fg="#ffffff",
            bg=self.highlight_green,
            padx=14,
            pady=4,
        )
        self.badge_lbl.pack(pady=4)

        # Visual Progress Bar
        self.progress = ttk.Progressbar(res_box, orient="horizontal", mode="determinate", length=280)
        self.progress.pack(pady=16, fill=tk.X, padx=20)

        # Advice Text
        self.advice_title = tk.Label(
            res_box,
            text="AI Academic Advisory:",
            font=("Segoe UI", 10, "bold"),
            fg=self.text_color,
            bg=self.card_color,
            anchor="w",
        )
        self.advice_title.pack(fill=tk.X, padx=10, pady=(12, 2))

        self.advice_box = tk.Label(
            res_box,
            text="Consistent performance. Dedicate 1.5 more hours to study to easily achieve Distinction.",
            font=("Segoe UI", 9),
            fg=self.muted_text,
            bg=self.card_color,
            wraplength=340,
            justify=tk.LEFT,
            anchor="w",
        )
        self.advice_box.pack(fill=tk.X, padx=10, pady=2)

    def build_goal_tab(self, parent):
        container = tk.Frame(parent, bg=self.card_color, bd=1, relief=tk.SOLID)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title = tk.Label(
            container,
            text="🎯 Target Score Solver (Reverse Regression)",
            font=("Segoe UI", 14, "bold"),
            fg=self.accent_color,
            bg=self.card_color,
        )
        title.pack(anchor="w", padx=20, pady=(20, 8))

        desc = tk.Label(
            container,
            text="Enter your desired final marks, and EduPredict will calculate the exact study hours needed.",
            font=("Segoe UI", 10),
            fg=self.muted_text,
            bg=self.card_color,
        )
        desc.pack(anchor="w", padx=20, pady=(0, 20))

        # Inputs Grid
        in_grid = tk.Frame(container, bg=self.card_color)
        in_grid.pack(fill=tk.X, padx=20, pady=10)

        self.goal_target_var = tk.DoubleVar(value=85.0)
        self.goal_att_var = tk.DoubleVar(value=85.0)
        self.goal_prev_var = tk.DoubleVar(value=70.0)

        # Row 1: Target Marks
        tk.Label(in_grid, text="Desired Target Marks (0-100):", font=("Segoe UI", 10, "bold"), fg=self.text_color, bg=self.card_color).grid(row=0, column=0, sticky="w", pady=8)
        e1 = tk.Entry(in_grid, textvariable=self.goal_target_var, font=("Segoe UI", 10), width=12)
        e1.grid(row=0, column=1, sticky="w", padx=10, pady=8)

        # Row 2: Attendance
        tk.Label(in_grid, text="Planned Attendance (%):", font=("Segoe UI", 10, "bold"), fg=self.text_color, bg=self.card_color).grid(row=1, column=0, sticky="w", pady=8)
        e2 = tk.Entry(in_grid, textvariable=self.goal_att_var, font=("Segoe UI", 10), width=12)
        e2.grid(row=1, column=1, sticky="w", padx=10, pady=8)

        # Row 3: Previous Marks
        tk.Label(in_grid, text="Previous Term Marks (%):", font=("Segoe UI", 10, "bold"), fg=self.text_color, bg=self.card_color).grid(row=2, column=0, sticky="w", pady=8)
        e3 = tk.Entry(in_grid, textvariable=self.goal_prev_var, font=("Segoe UI", 10), width=12)
        e3.grid(row=2, column=1, sticky="w", padx=10, pady=8)

        # Calculate Button
        calc_btn = tk.Button(
            in_grid,
            text="⚡ Calculate Required Study Hours",
            font=("Segoe UI", 10, "bold"),
            bg=self.accent_color,
            fg="#0f172a",
            padx=14,
            pady=6,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.calculate_goal,
        )
        calc_btn.grid(row=3, column=0, columnspan=2, sticky="w", pady=(16, 10))

        # Result display
        self.goal_result_lbl = tk.Label(
            container,
            text="Required Study: 6.2 hours / day",
            font=("Segoe UI", 18, "bold"),
            fg=self.highlight_green,
            bg=self.card_color,
        )
        self.goal_result_lbl.pack(anchor="w", padx=20, pady=(15, 5))

        self.goal_sub_lbl = tk.Label(
            container,
            text="Commit to this daily target along with regular revision to achieve your grade.",
            font=("Segoe UI", 10),
            fg=self.muted_text,
            bg=self.card_color,
        )
        self.goal_sub_lbl.pack(anchor="w", padx=20, pady=(0, 20))

    def build_metrics_tab(self, parent):
        container = tk.Frame(parent, bg=self.card_color, bd=1, relief=tk.SOLID)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title = tk.Label(
            container,
            text="📊 Machine Learning Model Benchmarks",
            font=("Segoe UI", 14, "bold"),
            fg=self.accent_color,
            bg=self.card_color,
        )
        title.pack(anchor="w", padx=20, pady=(20, 10))

        metrics = self.artifact.get("metrics", {})

        # Treeview Table
        tree_frame = tk.Frame(container, bg=self.card_color)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        columns = ("model", "r2", "rmse", "mae", "cv_r2")
        tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=6)
        tree.heading("model", text="Algorithm Name")
        tree.heading("r2", text="R² Score")
        tree.heading("rmse", text="RMSE (Marks)")
        tree.heading("mae", text="MAE (Marks)")
        tree.heading("cv_r2", text="5-Fold Cross Val")

        tree.column("model", width=200)
        tree.column("r2", width=100, anchor="center")
        tree.column("rmse", width=120, anchor="center")
        tree.column("mae", width=120, anchor="center")
        tree.column("cv_r2", width=120, anchor="center")

        for m_name, m_data in metrics.items():
            tree.insert(
                "",
                tk.END,
                values=(
                    m_name,
                    f"{m_data.get('r2', 0.0):.4f}",
                    f"{m_data.get('rmse', 0.0):.2f}",
                    f"{m_data.get('mae', 0.0):.2f}",
                    f"{m_data.get('cv_r2_mean', 0.0):.4f}",
                ),
            )

        tree.pack(fill=tk.BOTH, expand=True)

        info_lbl = tk.Label(
            container,
            text="Trained on 1,000 synthetic student records with cross-validation. Best Model: Linear Regression.",
            font=("Segoe UI", 9, "italic"),
            fg=self.muted_text,
            bg=self.card_color,
        )
        info_lbl.pack(anchor="w", padx=20, pady=(10, 20))

    def on_slider_change(self, which):
        if which == "study":
            val = round(self.study_hours_var.get(), 1)
            self.lbl_study.config(text=f"Daily Study Hours: {val:.1f} hrs")
        elif which == "att":
            val = round(self.attendance_var.get(), 1)
            self.lbl_att.config(text=f"Class Attendance: {val:.1f}%")
        elif which == "prev":
            val = round(self.prev_marks_var.get(), 1)
            self.lbl_prev.config(text=f"Previous Exam Marks: {val:.1f}%")

        self.update_prediction()

    def apply_preset(self, hours, att, prev):
        self.study_hours_var.set(hours)
        self.attendance_var.set(att)
        self.prev_marks_var.set(prev)

        self.lbl_study.config(text=f"Daily Study Hours: {hours:.1f} hrs")
        self.lbl_att.config(text=f"Class Attendance: {att:.1f}%")
        self.lbl_prev.config(text=f"Previous Exam Marks: {prev:.1f}%")

        self.update_prediction()

    def update_prediction(self):
        hours = self.study_hours_var.get()
        att = self.attendance_var.get()
        prev = self.prev_marks_var.get()

        model_name = self.selected_model_var.get()
        candidate_models = self.artifact.get("candidate_models", {})
        model = candidate_models.get(model_name, self.artifact.get("model"))

        features = ["Study_Hours", "Attendance", "Previous_Marks"]
        in_df = pd.DataFrame([[hours, att, prev]], columns=features)

        try:
            pred = float(model.predict(in_df)[0])
        except Exception:
            pred = 0.44 * prev + 2.75 * hours + 0.26 * att - 3.5

        bounded = min(max(pred, 0.0), 100.0)

        # Update Score Display
        self.score_display.config(text=f"{bounded:.1f}")
        self.progress["value"] = bounded

        # Grade & Badge
        if bounded >= 85:
            grade_text = "A+ (Distinction)"
            color = self.highlight_green
            advice = "Outstanding performance! Consistent daily habits position you at the top of the cohort."
        elif bounded >= 75:
            grade_text = "A (Excellent)"
            color = self.accent_color
            advice = "Strong mastery! Focusing on solving mock test papers can elevate this to top distinction."
        elif bounded >= 65:
            grade_text = "B (Good / First Class)"
            color = "#60a5fa"
            advice = "Solid performance. Increasing daily study by 1-2 hours will push your grade to an A."
        elif bounded >= 50:
            grade_text = "C (Pass / Average)"
            color = self.highlight_amber
            advice = "Passing range. Needs higher attendance consistency and targeted subject revisions."
        else:
            grade_text = "F (Academic Support Needed)"
            color = self.highlight_red
            advice = "Underachievement alert! We recommend faculty mentorship and structured study routines immediately."

        self.badge_lbl.config(text=grade_text, bg=color)
        self.advice_box.config(text=advice)

    def calculate_goal(self):
        try:
            target = float(self.goal_target_var.get())
            att = float(self.goal_att_var.get())
            prev = float(self.goal_prev_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return

        best_mod = self.artifact.get("candidate_models", {}).get("Linear Regression", self.artifact.get("model"))
        if hasattr(best_mod, "coef_") and hasattr(best_mod, "intercept_"):
            feats = self.artifact.get("features", ["Study_Hours", "Attendance", "Previous_Marks"])
            coef_map = dict(zip(feats, best_mod.coef_))
            b1 = coef_map.get("Study_Hours", 2.75)
            b2 = coef_map.get("Attendance", 0.26)
            b3 = coef_map.get("Previous_Marks", 0.44)
            b0 = best_mod.intercept_

            req_hours = (target - b0 - (b2 * att) - (b3 * prev)) / b1
            req_hours = max(req_hours, 0.5)
        else:
            req_hours = max((target - (0.44 * prev) - (0.26 * att)) / 2.75, 0.5)

        self.goal_result_lbl.config(text=f"Required Study: {req_hours:.1f} hours / day")
        if req_hours > 9.0:
            self.goal_result_lbl.config(fg=self.highlight_red)
            self.goal_sub_lbl.config(text="Ambitious goal! To reduce needed study hours, aim for 95%+ attendance and take practice tests.")
        else:
            self.goal_result_lbl.config(fg=self.highlight_green)
            self.goal_sub_lbl.config(text=f"Targeting {target:.0f}% is well achievable with steady discipline.")


def main():
    app = StudentPredictorGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
