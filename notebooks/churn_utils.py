"""Reusable data-cleaning code for the customer-churn notebooks (08–11).

Notebook 08 develops each of these steps one at a time and explains why.
Notebooks 09–11 import them from here, which is what you'd do in a real project:
once exploration settles on a cleaning recipe, it moves out of the notebook into a module
so every notebook, script and production job cleans data the same way.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).with_name("data")
RAW_PATH = DATA_DIR / "customer_churn.csv"

TARGET = "churned"
NUMERIC = ["age", "monthly_charges", "tenure_months", "total_charges", "support_calls",
           "avg_session_minutes", "satisfaction_score", "last_login_days_ago"]
CATEGORICAL = ["region", "plan", "contract", "payment_method", "has_partner"]
# Columns that must never be used as features:
#   customer_id          — an identifier, carries no general pattern
#   cancellation_reason  — only filled in AFTER a customer churns → target leakage
#   signup_date          — already summarised by tenure_months
DROP = ["customer_id", "cancellation_reason", "signup_date"]


def load_raw(path=RAW_PATH):
    """Read the raw CSV, regenerating it first if it's missing."""
    path = Path(path)
    if not path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("make_churn_data", DATA_DIR / "make_churn_data.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        mod.make().to_csv(path, index=False)
    return pd.read_csv(path)


def clean(df):
    """Fix the data-quality problems found in notebook 08. Returns a new DataFrame.

    Only row-by-row fixes happen here (no statistics learned from the data), so it is safe
    to run before the train/test split. Anything that LEARNS from data — imputing medians,
    scaling, encoding — belongs in a scikit-learn Pipeline fitted on the training set only.
    """
    df = df.copy()
    df = df.drop_duplicates()                                                  # duplicated export rows
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")  # text → number; " " → NaN
    df.loc[df["tenure_months"] == 0, "total_charges"] = 0.0                    # brand-new customers owe nothing yet
    df["region"] = df["region"].str.strip().str.lower()                        # "NORTH", "North " → "north"
    df.loc[(df["age"] < 18) | (df["age"] > 100), "age"] = np.nan              # impossible ages → missing
    df.loc[df["monthly_charges"] < 0, "monthly_charges"] = np.nan             # sign errors → missing
    return df.drop(columns=DROP).reset_index(drop=True)


def add_features(df):
    """Feature engineering from notebook 08 (row-by-row, so safe before splitting)."""
    df = df.copy()
    df["charges_per_month_of_tenure"] = df["total_charges"] / (df["tenure_months"] + 1)
    df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)
    df["satisfaction_missing"] = df["satisfaction_score"].isna().astype(int)
    df["log_last_login"] = np.log1p(df["last_login_days_ago"])
    return df


ENGINEERED = ["charges_per_month_of_tenure", "is_new_customer", "satisfaction_missing", "log_last_login"]


def load_clean(with_features=True):
    """Raw CSV → cleaned (and optionally feature-engineered) DataFrame, plus X and y."""
    df = clean(load_raw())
    if with_features:
        df = add_features(df)
    X, y = df.drop(columns=TARGET), df[TARGET].to_numpy()
    return df, X, y
