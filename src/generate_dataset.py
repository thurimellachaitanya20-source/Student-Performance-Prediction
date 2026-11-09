"""Synthetic dataset generator for Student Performance Prediction.
Generates realistic, statistically consistent student records for academic evaluation.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def generate_student_data(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)

    # 1. Study Hours (1 to 12 hours/day)
    study_hours = np.clip(np.random.normal(loc=5.5, scale=2.1, size=n_samples), 1.0, 12.0)
    study_hours = np.round(study_hours, 1)

    # 2. Attendance percentage (55% to 100%)
    attendance = np.clip(np.random.normal(loc=82.0, scale=9.5, size=n_samples), 55.0, 100.0)
    attendance = np.round(attendance, 1)

    # 3. Previous Marks (35 to 98)
    previous_marks = np.clip(np.random.normal(loc=70.0, scale=13.5, size=n_samples), 35.0, 98.0)
    previous_marks = np.round(previous_marks, 1)

    # 4. Sleep Hours (4 to 9.5 hours)
    sleep_hours = np.clip(np.random.normal(loc=7.0, scale=1.1, size=n_samples), 4.5, 9.5)
    sleep_hours = np.round(sleep_hours, 1)

    # 5. Practice Tests Taken (0 to 10)
    practice_tests = np.random.poisson(lam=4.2, size=n_samples)
    practice_tests = np.clip(practice_tests, 0, 10)

    # 6. Noise term
    noise = np.random.normal(loc=0.0, scale=2.8, size=n_samples)

    # 7. Final Marks calculation (realistic composite equation)
    # Base marks influenced primarily by previous foundation, study effort, attendance consistency
    final_marks = (
        0.44 * previous_marks
        + 2.75 * study_hours
        + 0.26 * attendance
        + 0.75 * practice_tests
        + 0.45 * (sleep_hours - 7.0)
        + noise
        - 3.5
    )
    final_marks = np.clip(final_marks, 25.0, 100.0)
    final_marks = np.round(final_marks, 1)

    # Assign Letter Grade based on Final Marks
    def get_grade(m: float) -> str:
        if m >= 90:
            return "A+ (Distinction)"
        if m >= 80:
            return "A (Excellent)"
        if m >= 70:
            return "B (Good)"
        if m >= 60:
            return "C (Satisfactory)"
        if m >= 50:
            return "D (Pass)"
        return "F (Needs Support)"

    grades = [get_grade(m) for m in final_marks]

    student_ids = [f"STU{1001 + i}" for i in range(n_samples)]

    df = pd.DataFrame(
        {
            "Student_ID": student_ids,
            "Study_Hours": study_hours,
            "Attendance": attendance,
            "Previous_Marks": previous_marks,
            "Sleep_Hours": sleep_hours,
            "Practice_Tests": practice_tests,
            "Final_Marks": final_marks,
            "Grade": grades,
        }
    )
    return df


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_student_data(n_samples=1000, random_seed=42)

    # Save full dataset
    full_path = DATA_DIR / "student_performance_dataset.csv"
    df.to_csv(full_path, index=False)
    print(f"Generated {len(df)} samples saved to: {full_path.name}")

    # Also save standard core csv matching the baseline columns
    core_cols = ["Study_Hours", "Attendance", "Previous_Marks", "Sleep_Hours", "Practice_Tests", "Final_Marks"]
    core_path = DATA_DIR / "student_performance.csv"
    df[core_cols].to_csv(core_path, index=False)
    print(f"Updated {core_path.name} with {len(df)} realistic records.")


if __name__ == "__main__":
    main()
