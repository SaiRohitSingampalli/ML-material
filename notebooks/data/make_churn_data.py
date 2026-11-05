"""Generate the synthetic 'customer churn' dataset used in notebooks 08–11.

It imitates a real subscription business's customer table, with deliberately messy data
(missing values, inconsistent text, impossible values, duplicate rows, numbers stored as text,
and one leaky column) so the notebooks can practise cleaning it.

The churn label comes from a known formula, which lets the notebooks check that models and
explanations recover the true drivers. Run from the notebooks folder:
    python data/make_churn_data.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

def make(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    signup = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.integers(0, 1461, n), unit="D")
    snapshot = pd.Timestamp("2025-01-01")
    tenure = ((snapshot - signup).days / 30.44).astype(int)

    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.45, 0.35, 0.20])
    base_price = pd.Series(plan).map({"basic": 35, "standard": 60, "premium": 95}).to_numpy()
    monthly = np.round(base_price + rng.normal(0, 8, n) + (contract == "month-to-month") * 5, 2)
    age = np.clip(rng.normal(42, 13, n), 18, 85).round().astype(int)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.3, 0.25, 0.25, 0.2])
    payment = rng.choice(["credit card", "bank transfer", "electronic check", "mailed check"], n, p=[0.35, 0.25, 0.3, 0.1])
    partner = rng.choice(["yes", "no"], n, p=[0.45, 0.55])
    support_calls = rng.poisson(1.2 + (plan == "basic") * 0.8, n)
    session_min = np.round(rng.gamma(4, 6, n) + (plan == "premium") * 10, 1)
    satisfaction = np.clip(np.round(4 - 0.35 * support_calls + rng.normal(0, 0.8, n)), 1, 5)
    last_login = np.round(rng.exponential(8, n) + (satisfaction <= 2) * rng.exponential(15, n)).astype(int)

    # True churn mechanism (log-odds), including a non-linear effect and an interaction
    logit = (-1.6
             + 1.4 * (contract == "month-to-month") - 1.1 * (contract == "two-year")
             - 0.045 * np.minimum(tenure, 36)
             + 0.30 * support_calls
             - 0.45 * (satisfaction - 3)
             + 0.035 * np.minimum(last_login, 40)
             + 0.018 * (monthly - 60)
             + 0.55 * (payment == "electronic check")
             + 0.7 * ((monthly > 80) & (tenure < 12))          # interaction: pricey + new
             - 0.25 * (partner == "yes")
             + rng.normal(0, 0.3, n))
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "age": age, "region": region, "plan": plan, "contract": contract,
        "payment_method": payment, "has_partner": partner,
        "monthly_charges": monthly, "tenure_months": tenure,
        "total_charges": np.round(monthly * tenure + rng.normal(0, 5, n), 2),
        "support_calls": support_calls, "avg_session_minutes": session_min,
        "satisfaction_score": satisfaction, "last_login_days_ago": last_login,
        "cancellation_reason": np.where(churned == 1, rng.choice(["price", "service", "competitor", "moved"], n), None),
        "churned": churned,
    })

    # ---- make it messy, like real data ----
    # unhappy customers are more likely to skip the survey → missingness itself carries signal
    skip = rng.random(n) < (0.08 + 0.30 * (satisfaction <= 2))
    df.loc[skip, "satisfaction_score"] = np.nan
    df.loc[rng.random(n) < 0.04, "avg_session_minutes"] = np.nan
    df.loc[rng.random(n) < 0.03, "age"] = np.nan
    messy = rng.random(n)
    df.loc[messy < 0.05, "region"] = df.loc[messy < 0.05, "region"].str.upper()           # "NORTH"
    df.loc[(messy >= 0.05) & (messy < 0.08), "region"] = df.loc[(messy >= 0.05) & (messy < 0.08), "region"].str.title() + " "  # "North "
    df["total_charges"] = df["total_charges"].astype(str)
    df.loc[tenure == 0, "total_charges"] = " "                           # new customers: blank text
    bad = rng.choice(n, 8, replace=False)
    df.loc[bad[:5], "age"] = 999                                         # data-entry errors
    df.loc[bad[5:], "monthly_charges"] *= -1                             # sign errors
    dups = df.sample(60, random_state=seed)                              # duplicated export rows
    df = pd.concat([df, dups]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    out = Path(__file__).with_name("customer_churn.csv")
    d = make()
    d.to_csv(out, index=False)
    print(f"wrote {out} with {len(d)} rows; churn rate {d['churned'].mean():.3f}")
