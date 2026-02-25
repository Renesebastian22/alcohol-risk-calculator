import streamlit as st
import pandas as pd
import numpy as np

# =====================================================
# SETTINGS
# =====================================================

COHORT_MIN_AGE = 15
COHORT_MAX_AGE = 79
YEARS = 5

DISEASES = [
    "breast_cancer",
    "cirrhosis",
    "esophageal_cancer",
    "epilepsy",
    "injuries",
    "ischaemic_stroke",
    "aud",
    "liver_cancer",
    "ischemic_heart_disease",
    "pancreatitis",
]

# =====================================================
# ⚠️ REPLACE WITH YOUR REAL RR FUNCTION
# =====================================================

# --- 3-level anchors (everything except AUD uses these) ---
RR_THREE_LEVEL = {
    "cirrhosis":              ({"1": 2.90, "2": 7.13, "3": 26.52}, {"1": 2.90, "2": 7.13, "3": 26.52}),
    "pancreatitis":           ({"1": 1.34, "2": 1.78, "3": 3.19},  {"1": 1.34, "2": 1.78, "3": 3.19}),
    "intracerebral_stroke":   ({"1": 1.19, "2": 1.82, "3": 4.70},  {"1": 1.19, "2": 1.82, "3": 4.70}),
    "ischaemic_stroke":       ({"1": 0.90, "2": 1.17, "3": 4.37},  {"1": 0.90, "2": 1.17, "3": 4.37}),
    "epilepsy":               ({"1": 1.34, "2": 7.22, "3": 7.52},  {"1": 1.23, "2": 7.52, "3": 6.83}),
    "injuries":               ({"1": 1.12, "2": 1.26, "3": 1.58},  {"1": 1.12, "2": 1.26, "3": 1.58}),
    "pharynx_cancer":         ({"1": 1.86, "2": 3.11, "3": 6.45},  {"1": 1.86, "2": 3.11, "3": 6.45}),
    "liver_cancer":           ({"1": 1.19, "2": 1.40, "3": 1.81},  {"1": 1.19, "2": 1.40, "3": 1.81}),
    "esophageal_cancer":      ({"1": 1.39, "2": 1.93, "3": 3.59},  {"1": 1.39, "2": 1.93, "3": 3.59}),
    "breast_cancer":          ({"1": 1.25, "2": 1.55, "3": 2.41},  {"1": 1.25, "2": 1.55, "3": 2.41}),
}

def _rr_loglinear(cons, rr1, rr2, rr3):
    c = np.clip(np.asarray(cons, float), 0.0, 100.0)
    rr_vals = np.array([1.0, float(rr1), float(rr2), float(rr3)], dtype=float)
    rr_vals = np.where(rr_vals <= 0, 1.0, rr_vals)
    logv = np.log(rr_vals)

    out = np.empty_like(c, float)

    m0 = (c <= 25.0)
    if np.any(m0):
        t = c[m0] / 25.0
        out[m0] = np.exp(logv[0] + t * (logv[1] - logv[0]))

    m1 = (c > 25.0) & (c <= 50.0)
    if np.any(m1):
        t = (c[m1] - 25.0) / 25.0
        out[m1] = np.exp(logv[1] + t * (logv[2] - logv[1]))

    m2 = (c > 50.0)
    if np.any(m2):
        t = (c[m2] - 50.0) / 50.0
        out[m2] = np.exp(logv[2] + t * (logv[3] - logv[2]))

    return out

def get_rr_array(disease, sex_arr, cons_arr):
    cons_arr = np.asarray(cons_arr, float)

    # AUD uses quadratic
    if disease == "aud":
        return rr_aud_incidence(cons_arr)

    # default: neutral RR if disease not implemented (prevents crash)
    rr = np.ones_like(cons_arr, float)

    anchors = RR_THREE_LEVEL.get(disease)
    if anchors is None:
        return rr

    rr_f, rr_m = anchors
    sx = np.char.title(np.asarray(sex_arr, str))

    mF = (sx == "Female")
    if np.any(mF):
        rr[mF] = _rr_loglinear(cons_arr[mF], rr_f["1"], rr_f["2"], rr_f["3"])

    mM = (sx == "Male")
    if np.any(mM):
        rr[mM] = _rr_loglinear(cons_arr[mM], rr_m["1"], rr_m["2"], rr_m["3"])

    return rr



# --- AUD incidence quadratic (fixed coefficients from eMethods) ---
AUD_INC_B1 = 0.0517
AUD_INC_B2 = -0.000064498319

def rr_aud_incidence(cons_arr):
    # Cap exposure for the RR mapping at 100 g/day ethanol
    x = np.clip(np.asarray(cons_arr, float), 0.0, 100.0)
    rr = np.exp(AUD_INC_B1 * x + AUD_INC_B2 * (x ** 2))
    return np.maximum(rr, 1.0)

def aud_eligible_100(cons_arr):
    q = np.asarray(cons_arr, float)
    return (q > 0.0) & (q <= 100.0)


# =====================================================
# LOAD DATA (NO CACHE)
# =====================================================

inc = pd.read_csv("incidence_age_sex_15_79_wide.csv")
denom = pd.read_csv("denom_rr_age_sex_15_79_wide.csv")

inc["sex"] = inc["sex"].astype(str).str.title()
denom["sex"] = denom["sex"].astype(str).str.title()

base = inc.merge(denom, on=["age", "sex"], how="inner")
base["age"] = base["age"].astype(int)

lookup = {
    (int(r.age), str(r.sex)): r
    for r in base.itertuples(index=False)
}

# =====================================================
# ENGINE
# =====================================================

def one_year_prob(disease, age, sex, consumption):

    if age < COHORT_MIN_AGE or age > COHORT_MAX_AGE:
        return 0.0

    row = lookup.get((int(age), sex))
    if row is None:
        return 0.0

    p_m = float(getattr(row, f"inc_{disease}"))
    denom_val = float(getattr(row, f"denom_{disease}"))

    if denom_val <= 0 or p_m <= 0:
        return 0.0

    lambda_m = -np.log(1 - p_m)
    lambda_0 = lambda_m / denom_val

    rr = float(
        get_rr_array(
            disease,
            np.array([sex]),
            np.array([float(consumption)])
        )[0]
    )

    if not np.isfinite(rr) or rr <= 0:
        rr = 1.0

    lambda_z = lambda_0 * rr
    p = 1 - np.exp(-lambda_z)

    return max(0.0, min(p, 0.999))


def five_year_risk(disease, age, sex, consumption):

    survival = 1.0

    for k in range(YEARS):
        p = one_year_prob(disease, age + k, sex, consumption)
        survival *= (1 - p)

    return 1 - survival


def five_year_alcohol_effect(disease, age, sex, consumption):

    risk_exposed = five_year_risk(disease, age, sex, consumption)
    risk_cf = five_year_risk(disease, age, sex, 0.0)

    excess = max(0.0, risk_exposed - risk_cf)
    af = excess / risk_exposed if risk_exposed > 0 else 0.0
    rr = risk_exposed / risk_cf if risk_cf > 0 else np.nan

    return risk_exposed, risk_cf, excess, af, rr

# =====================================================
# UI
# =====================================================

st.title("5-Year Alcohol-Related Disease Risk Calculator")

age = st.slider("Age", COHORT_MIN_AGE, COHORT_MAX_AGE, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
consumption = st.slider("Alcohol consumption (grams per day)", 0, 120, 30)

if st.button("Calculate Risk"):

    results = []

    for d in DISEASES:

        r_exp, r_cf, excess, af, rr = five_year_alcohol_effect(
            d, age, sex, consumption
        )

        results.append({
            "Disease": d.replace("_", " ").title(),
            "5y Risk (%)": round(100 * r_exp, 3),
            "5y Risk (0 g/day) (%)": round(100 * r_cf, 3),
            "Excess Risk (%)": round(100 * excess, 3),
            "Attributable Fraction (%)": round(100 * af, 2),
            "Risk Ratio": round(rr, 3) if not np.isnan(rr) else "-"
        })

    df = pd.DataFrame(results)

    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    st.subheader("Alcohol-Attributable Excess Risk")
    st.bar_chart(df.set_index("Disease")["Excess Risk (%)"])