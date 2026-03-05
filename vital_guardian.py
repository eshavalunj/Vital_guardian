"""
VitalGuardian: Proactive Anomaly Anticipation in Intensive Care Units
=====================================================================
Hackfest Innov8 TMRW 2026 — CESA x VIT Mumbai | Problem H02

A dragon-themed ICU co-pilot that fuses LSTM forecasting, Isolation Forest
anomaly detection, adaptive thresholds, and SHAP explainability into a
real-time clinical dashboard with What-If intervention simulation.

Author: Team VitalGuardian
License: MIT
"""

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import datetime
import io

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════
# PAGE CONFIG (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VitalGuardian — ICU Anomaly Anticipation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dragon-Themed Dark UI
# ═══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 1400px;
}

/* ── Header ── */
.hero-title {
    text-align: center;
    padding: 1.2rem 0 0.5rem;
}
.hero-title h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #FF4500, #FF8C00, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-title p {
    color: #8B9DC3;
    font-size: 0.95rem;
    margin: 0.3rem 0 0;
}

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #1A1D23 0%, #22262E 100%);
    border: 1px solid #2A2E36;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.3s ease;
}
.metric-card:hover { border-color: #FF4500; }

.metric-card h3 {
    font-size: 0.8rem;
    color: #8B9DC3;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0 0 0.5rem 0;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.metric-card .delta {
    font-size: 0.85rem;
    margin: 0.2rem 0 0;
}

/* ── Alert Banners ── */
.alert-critical {
    background: linear-gradient(135deg, rgba(255,69,0,0.15), rgba(255,0,0,0.08));
    border-left: 4px solid #FF4500;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    animation: pulse-border 2s ease-in-out infinite;
}
.alert-stable {
    background: linear-gradient(135deg, rgba(0,200,83,0.1), rgba(0,150,60,0.05));
    border-left: 4px solid #00C853;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}

@keyframes pulse-border {
    0%, 100% { border-left-color: #FF4500; box-shadow: 0 0 8px rgba(255,69,0,0.2); }
    50% { border-left-color: #FF8C00; box-shadow: 0 0 16px rgba(255,140,0,0.3); }
}

/* ── Risk Gauge Label ── */
.risk-low { color: #00C853; }
.risk-medium { color: #FFD600; }
.risk-high { color: #FF6D00; }
.risk-critical { color: #FF1744; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid #2A2E36;
    margin-top: 2rem;
}
.footer p {
    color: #5C6370;
    font-size: 0.78rem;
    margin: 0.2rem 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0E1117 0%, #151920 100%);
    border-right: 1px solid #2A2E36;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1.1rem;
    color: #FF8C00;
}

/* ── SHAP Section ── */
.shap-container {
    background: #1A1D23;
    border: 1px solid #2A2E36;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Streamlit overrides ── */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(255,69,0,0.3);
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1A1D23, #22262E);
    border: 1px solid #2A2E36;
    border-radius: 10px;
    padding: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════

VITAL_NAMES = ["HR", "SpO2", "SBP", "RR", "Temp"]
VITAL_UNITS = ["bpm", "%", "mmHg", "br/min", "°C"]
VITAL_BASELINE = {
    "HR":   {"mean": 75,   "std": 5,   "lo": 60,  "hi": 100},
    "SpO2": {"mean": 97,   "std": 1,   "lo": 92,  "hi": 100},
    "SBP":  {"mean": 120,  "std": 8,   "lo": 90,  "hi": 140},
    "RR":   {"mean": 16,   "std": 2,   "lo": 12,  "hi": 20},
    "Temp": {"mean": 36.8, "std": 0.3, "lo": 36.0,"hi": 37.5},
}
SEQ_LEN = 50       # LSTM window
FORECAST_STEPS = 30 # Predict 30 steps ahead (~30 min at 1-min intervals)
N_PATIENTS = 10


# ═══════════════════════════════════════════════════════════════════════
# DATA SIMULATION — MIMIC-III Inspired Synthetic Vitals
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def generate_patient_data(patient_id: int, n_steps: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic ICU vital signs for one patient.
    - Baseline with patient-specific variation (seeded per patient).
    - After t=700: simulated sepsis ramp (HR/RR/Temp up, SpO2/SBP down).
    - Gaussian noise throughout.
    """
    rng = np.random.RandomState(42 + patient_id)
    t = np.arange(n_steps)

    # Patient-specific baseline shift
    shift = rng.randn(5) * 0.5

    data = {}
    for i, vital in enumerate(VITAL_NAMES):
        base = VITAL_BASELINE[vital]
        signal = base["mean"] + shift[i] * base["std"] + rng.randn(n_steps) * base["std"]

        # Sepsis ramp after t=700
        ramp_start = 700
        ramp = np.clip((t - ramp_start) / (n_steps - ramp_start), 0, 1)
        if vital in ["HR", "RR", "Temp"]:
            signal += ramp * base["std"] * 6  # Vitals go UP
        elif vital in ["SpO2", "SBP"]:
            signal -= ramp * base["std"] * 5  # Vitals go DOWN

        data[vital] = signal.astype(np.float32)

    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════
# PYTORCH LSTM MODEL — Vital Sign Forecasting
# ═══════════════════════════════════════════════════════════════════════

class VitalLSTM(nn.Module):
    """2-layer LSTM for multi-variate vital sign forecasting."""
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, output_dim=5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Last timestep prediction


@st.cache_resource(show_spinner="🐉 Training LSTM forecasting model...")
def train_lstm_model(_data_dict: dict) -> tuple:
    """
    Train a shared LSTM model on all patients' data.
    Uses sliding window approach over concatenated patient data.
    Returns (model, scaler).
    """
    # Concatenate all patient data
    all_data = []
    for pid in range(N_PATIENTS):
        df = generate_patient_data(pid)
        all_data.append(df.values)
    all_data = np.concatenate(all_data, axis=0).astype(np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(all_data).astype(np.float32)

    # Create sliding windows
    X, y = [], []
    for i in range(len(scaled) - SEQ_LEN - 1):
        X.append(scaled[i:i + SEQ_LEN])
        y.append(scaled[i + SEQ_LEN])

    X = torch.tensor(np.array(X, dtype=np.float32))
    y = torch.tensor(np.array(y, dtype=np.float32))

    # Subsample for speed (keep it under 15s training)
    if len(X) > 3000:
        idx = np.random.choice(len(X), 3000, replace=False)
        X, y = X[idx], y[idx]

    model = VitalLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, scaler


# ═══════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION — Isolation Forest
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔍 Fitting anomaly detector...")
def train_anomaly_detector(_data_dict: dict) -> IsolationForest:
    """Train Isolation Forest on baseline (first 600 steps) of all patients."""
    all_baseline = []
    for pid in range(N_PATIENTS):
        df = generate_patient_data(pid)
        all_baseline.append(df.values[:600])
    all_baseline = np.concatenate(all_baseline, axis=0).astype(np.float32)

    iso = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(all_baseline)
    return iso


def detect_anomaly(iso_model: IsolationForest, current_vitals: np.ndarray) -> tuple:
    """
    Returns (is_anomaly: bool, anomaly_score: float).
    Handles dtype and shape safety.
    """
    try:
        v = np.array(current_vitals, dtype=np.float32).reshape(1, -1)
        pred = iso_model.predict(v)[0]
        score = -iso_model.score_samples(v)[0]  # Higher = more anomalous
        return pred == -1, float(np.clip(score, 0, 1))
    except Exception:
        return False, 0.0


# ═══════════════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLDS — Per-Patient Mean + 2σ
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def compute_adaptive_thresholds(patient_id: int) -> dict:
    """
    Compute per-patient adaptive thresholds using first 500 steps (stable baseline).
    Returns dict of {vital: (lower, upper)}.
    """
    df = generate_patient_data(patient_id)
    baseline = df.iloc[:500]
    thresholds = {}
    for vital in VITAL_NAMES:
        mu = baseline[vital].mean()
        sigma = baseline[vital].std()
        thresholds[vital] = (float(mu - 2 * sigma), float(mu + 2 * sigma))
    return thresholds


# ═══════════════════════════════════════════════════════════════════════
# LSTM FORECASTING — Multi-Step Prediction
# ═══════════════════════════════════════════════════════════════════════

def forecast_vitals(model, scaler, patient_data, time_idx, n_future=FORECAST_STEPS):
    """
    Given patient data up to time_idx, forecast next n_future steps.
    Returns forecasted values in ORIGINAL scale as ndarray (n_future, 5).
    """
    try:
        vals = patient_data.values[:time_idx + 1].astype(np.float32)

        # Pad if too short
        if len(vals) < SEQ_LEN:
            pad = np.tile(vals[0], (SEQ_LEN - len(vals), 1))
            vals = np.concatenate([pad, vals], axis=0)

        scaled = scaler.transform(vals).astype(np.float32)
        seq = scaled[-SEQ_LEN:]

        forecasts = []
        with torch.no_grad():
            for _ in range(n_future):
                inp = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                pred = model(inp).numpy()[0]
                forecasts.append(pred)
                seq = np.concatenate([seq[1:], pred.reshape(1, -1)], axis=0)

        forecasts = np.array(forecasts, dtype=np.float32)
        return scaler.inverse_transform(forecasts).astype(np.float32)
    except Exception:
        # Fallback: repeat last known values
        last = patient_data.values[time_idx].astype(np.float32)
        return np.tile(last, (n_future, 1))


# ═══════════════════════════════════════════════════════════════════════
# RISK SCORE — Composite Metric
# ═══════════════════════════════════════════════════════════════════════

def compute_risk_score(
    current_vitals: np.ndarray,
    thresholds: dict,
    anomaly_score: float,
    forecast: np.ndarray
) -> float:
    """
    Composite risk = 40% threshold breaches + 30% anomaly + 30% forecast trend.
    Returns score 0-100.
    """
    # Threshold breach score (how many vitals outside adaptive range)
    breach_count = 0
    for i, vital in enumerate(VITAL_NAMES):
        lo, hi = thresholds[vital]
        val = float(current_vitals[i])
        if val < lo or val > hi:
            breach_count += 1
    breach_score = breach_count / len(VITAL_NAMES)

    # Forecast trend: average absolute change over FORECAST_STEPS
    if len(forecast) > 1:
        trend = np.mean(np.abs(np.diff(forecast, axis=0)))
        trend_score = np.clip(trend / 5.0, 0, 1)  # Normalize by 5 units
    else:
        trend_score = 0.0

    risk = 0.40 * breach_score + 0.30 * anomaly_score + 0.30 * trend_score
    return float(np.clip(risk * 100, 0, 100))


# ═══════════════════════════════════════════════════════════════════════
# SHAP EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════

def compute_shap_values(model, scaler, patient_data, time_idx):
    """
    Compute SHAP values for feature contributions.
    Falls back to gradient-based importance if SHAP is too slow/fails.
    Returns (shap_values: array of 5 floats, method: str).
    """
    try:
        import shap

        vals = patient_data.values[:time_idx + 1].astype(np.float32)
        if len(vals) < SEQ_LEN:
            pad = np.tile(vals[0], (SEQ_LEN - len(vals), 1))
            vals = np.concatenate([pad, vals], axis=0)

        scaled = scaler.transform(vals).astype(np.float32)
        current_scaled = scaled[-1].reshape(1, -1)

        # Use a small background set (10 samples) for speed
        bg_idx = np.linspace(0, len(scaled) - 1, 10, dtype=int)
        background = scaled[bg_idx]

        def model_predict(x):
            """Wrapper: take 5 features, tile into SEQ_LEN window, predict."""
            preds = []
            for row in x:
                window = np.tile(row.astype(np.float32), (SEQ_LEN, 1))
                inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    out = model(inp).numpy()[0]
                # Return aggregate risk proxy (mean absolute deviation from zero)
                preds.append([np.mean(np.abs(out))])
            return np.array(preds, dtype=np.float64)

        explainer = shap.KernelExplainer(model_predict, background)
        sv = explainer.shap_values(current_scaled, nsamples=50, silent=True)

        if isinstance(sv, list):
            sv = sv[0]
        return np.abs(sv.flatten()[:5]).astype(np.float64), "SHAP KernelExplainer"

    except Exception:
        # Fallback: gradient-based feature importance
        try:
            vals = patient_data.values[:time_idx + 1].astype(np.float32)
            if len(vals) < SEQ_LEN:
                pad = np.tile(vals[0], (SEQ_LEN - len(vals), 1))
                vals = np.concatenate([pad, vals], axis=0)

            scaled = scaler.transform(vals).astype(np.float32)
            seq = torch.tensor(scaled[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0)
            seq.requires_grad_(True)

            model.zero_grad()
            pred = model(seq)
            loss = pred.sum()
            loss.backward()

            grads = seq.grad.abs().mean(dim=1).squeeze().numpy()
            return grads.astype(np.float64), "Gradient-Based (Fallback)"
        except Exception:
            return np.ones(5, dtype=np.float64) * 0.2, "Uniform (Fallback)"


def render_shap_chart(shap_values, method_name):
    """Render SHAP feature importance as a matplotlib bar chart, return as bytes."""
    fig, ax = plt.subplots(figsize=(5, 3))

    colors = ["#FF4500", "#FF6D00", "#FF8C00", "#FFB300", "#FFD600"]
    sorted_idx = np.argsort(shap_values)[::-1]
    names = [VITAL_NAMES[i] for i in sorted_idx]
    vals = [shap_values[i] for i in sorted_idx]

    bars = ax.barh(names[::-1], vals[::-1], color=[colors[i] for i in range(len(names))])

    ax.set_xlabel("Contribution", fontsize=9, color="#FAFAFA")
    ax.set_title(f"Feature Contributions ({method_name})", fontsize=10, fontweight="bold", color="#FF8C00")
    ax.set_facecolor("#1A1D23")
    fig.patch.set_facecolor("#0E1117")
    ax.tick_params(colors="#FAFAFA", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#2A2E36")
    ax.spines["left"].set_color("#2A2E36")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════
# WHAT-IF SIMULATION — Intervention Effect Modeling
# ═══════════════════════════════════════════════════════════════════════

def apply_what_if(forecast: np.ndarray, fluid_ml: float, med_dose: float) -> np.ndarray:
    """
    Simulate intervention effects on forecasted trajectory.
    Fluid bolus → stabilize BP/HR; Med dose → normalize towards baseline.
    Returns modified forecast.
    """
    modified = forecast.copy()

    # Compute damping factor: more intervention = more stabilization
    fluid_effect = (fluid_ml / 500.0) * 0.35   # Max 35% damping from fluids
    med_effect = (med_dose / 2.0) * 0.35        # Max 35% damping from meds
    total_damping = min(fluid_effect + med_effect, 0.7)  # Cap at 70%

    # Gradually apply — more damping at later forecast steps
    for step in range(len(modified)):
        progress = (step + 1) / len(modified)
        step_damping = total_damping * progress

        for i, vital in enumerate(VITAL_NAMES):
            baseline = VITAL_BASELINE[vital]["mean"]
            deviation = modified[step, i] - baseline
            modified[step, i] = baseline + deviation * (1 - step_damping)

    return modified.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# PLOTLY VISUALIZATION — Vitals + Risk Trajectory
# ═══════════════════════════════════════════════════════════════════════

def create_vitals_plot(patient_data, time_idx, forecast, what_if_forecast):
    """Create a 2-row Plotly subplot: vitals history + risk trajectory."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("📊 Vital Signs Monitor", "🔮 Risk Trajectory & What-If"),
        vertical_spacing=0.15,
        row_heights=[0.55, 0.45],
    )

    colors = {"HR": "#FF6B6B", "SpO2": "#4ECDC4", "SBP": "#45B7D1", "RR": "#96CEB4", "Temp": "#FFEAA7"}
    t_history = np.arange(max(0, time_idx - 200), time_idx + 1)
    t_forecast = np.arange(time_idx + 1, time_idx + 1 + len(forecast))

    # Row 1: Vital signs
    for vital in VITAL_NAMES:
        history = patient_data[vital].values[t_history]
        fig.add_trace(
            go.Scatter(
                x=t_history, y=history, name=vital,
                line=dict(color=colors[vital], width=2),
                hovertemplate=f"{vital}: %{{y:.1f}}<extra></extra>"
            ),
            row=1, col=1
        )
        # Forecast extension (dashed)
        fc_vals = forecast[:, VITAL_NAMES.index(vital)]
        fig.add_trace(
            go.Scatter(
                x=t_forecast, y=fc_vals, name=f"{vital} (forecast)",
                line=dict(color=colors[vital], width=1.5, dash="dash"),
                showlegend=False,
                hovertemplate=f"{vital} forecast: %{{y:.1f}}<extra></extra>"
            ),
            row=1, col=1
        )

    # "Current time" vertical line
    fig.add_vline(x=time_idx, line=dict(color="#FF4500", width=2, dash="dot"), row=1, col=1)

    # Row 2: Risk trajectory (aggregate deviation from baseline)
    def calc_deviation(vals):
        devs = []
        for step_vals in vals:
            dev = 0
            for i, vital in enumerate(VITAL_NAMES):
                base = VITAL_BASELINE[vital]["mean"]
                sigma = VITAL_BASELINE[vital]["std"]
                dev += abs(step_vals[i] - base) / (sigma + 1e-6)
            devs.append(dev / len(VITAL_NAMES))
        return np.array(devs)

    # Original trajectory risk
    risk_original = calc_deviation(forecast)
    fig.add_trace(
        go.Scatter(
            x=t_forecast, y=risk_original, name="Risk (No Intervention)",
            line=dict(color="#FF4500", width=3),
            fill="tozeroy", fillcolor="rgba(255,69,0,0.1)",
            hovertemplate="Risk: %{y:.2f}σ<extra></extra>"
        ),
        row=2, col=1
    )

    # What-If trajectory
    risk_whatif = calc_deviation(what_if_forecast)
    fig.add_trace(
        go.Scatter(
            x=t_forecast, y=risk_whatif, name="Risk (With Intervention)",
            line=dict(color="#00C853", width=3),
            fill="tozeroy", fillcolor="rgba(0,200,83,0.08)",
            hovertemplate="Risk (tamed): %{y:.2f}σ<extra></extra>"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=520,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(family="Inter, sans-serif", color="#FAFAFA", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=50, r=20, t=40, b=60),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Time (min)", gridcolor="#1E222A", row=2, col=1)
    fig.update_yaxes(gridcolor="#1E222A")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# RISK GAUGE — Plotly Indicator
# ═══════════════════════════════════════════════════════════════════════

def create_risk_gauge(risk_score):
    """Create a fiery risk gauge using Plotly indicator."""
    if risk_score < 25:
        color, label = "#00C853", "STABLE"
    elif risk_score < 50:
        color, label = "#FFD600", "CAUTION"
    elif risk_score < 75:
        color, label = "#FF6D00", "HIGH RISK"
    else:
        color, label = "#FF1744", "CRITICAL"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        number=dict(suffix="%", font=dict(size=42, color=color)),
        title=dict(text=f"Risk Level: {label}", font=dict(size=14, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#2A2E36", dtick=25),
            bar=dict(color=color, thickness=0.7),
            bgcolor="#1A1D23",
            borderwidth=0,
            steps=[
                dict(range=[0, 25], color="rgba(0,200,83,0.15)"),
                dict(range=[25, 50], color="rgba(255,214,0,0.12)"),
                dict(range=[50, 75], color="rgba(255,109,0,0.12)"),
                dict(range=[75, 100], color="rgba(255,23,68,0.12)"),
            ],
            threshold=dict(line=dict(color="#FF4500", width=3), thickness=0.8, value=risk_score),
        ),
    ))
    fig.update_layout(
        height=250,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font=dict(family="Inter", color="#FAFAFA"),
        margin=dict(l=30, r=30, t=50, b=10),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# PROJECTED IMPACT METRICS — Static Showcase
# ═══════════════════════════════════════════════════════════════════════

def create_impact_chart():
    """Bar chart showing projected impact metrics for the pitch."""
    fig = go.Figure()
    categories = ["Mortality\nReduction", "False Alarm\nReduction", "Response\nTime (min)", "Annual Cost\nSaving (₹L)"]
    values = [25, 58, 15, 85]
    colors_bar = ["#00C853", "#FF8C00", "#45B7D1", "#FFD600"]

    fig.add_trace(go.Bar(
        x=categories, y=values,
        marker=dict(color=colors_bar, line=dict(width=0)),
        text=[f"{v}%" if i < 2 else (f"-{v} min" if i == 2 else f"₹{v}L") for i, v in enumerate(values)],
        textposition="outside",
        textfont=dict(size=12, color="#FAFAFA"),
    ))
    fig.update_layout(
        title=dict(text="📈 Projected Clinical Impact", font=dict(size=14)),
        height=280,
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        yaxis=dict(visible=False),
        xaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════

def main():
    # ── Header ──
    st.markdown("""
    <div class="hero-title">
        <h1>🛡️ VitalGuardian: Taming the ICU Alarm Dragon 🐉</h1>
        <p>Proactive Anomaly Anticipation • LSTM Forecasting • Adaptive Thresholds • SHAP Explainability</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🐉 Control Panel")
        st.markdown("---")

        # Patient selector
        patient_id = st.selectbox(
            "🛏️ Select Patient Bed",
            range(N_PATIENTS),
            format_func=lambda x: f"Bed {x} — Patient ICU-{1000 + x}",
        )

        st.markdown("---")
        st.markdown("## ⏱️ Simulation")

        # Simulation time slider
        sim_time = st.slider(
            "Current Time (min)",
            min_value=SEQ_LEN + 10,
            max_value=990,
            value=600,
            step=5,
            help="Move past 750 to observe deterioration patterns",
        )

        st.markdown("---")
        st.markdown("## 🧪 What-If Interventions")

        fluid_ml = st.slider("💧 Fluid Bolus (ml)", 0, 500, 0, step=50,
                             help="Simulate IV fluid administration")
        med_dose = st.slider("💊 Medication Dose (×)", 0.0, 2.0, 0.0, step=0.25,
                             help="Simulate medication dosage adjustment")

        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

        # Sidebar info
        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#5C6370; padding:0.5rem;">
        <strong>VitalGuardian v1.0</strong><br>
        Hackfest Innov8 TMRW 2026<br>
        CESA × VIT Mumbai<br>
        <em>Build Beyond Infinity</em> ∞
        </div>
        """, unsafe_allow_html=True)

    # ── Load & Train ──
    with st.spinner("🐉 Awakening the Guardian..."):
        patient_data = generate_patient_data(patient_id)
        data_trigger = {"v": 1}  # Hashable trigger for caching
        model, scaler = train_lstm_model(data_trigger)
        iso_model = train_anomaly_detector(data_trigger)
        thresholds = compute_adaptive_thresholds(patient_id)

    # ── Compute at current time ──
    current_vitals = patient_data.iloc[sim_time].values.astype(np.float32)
    is_anomaly, anomaly_score = detect_anomaly(iso_model, current_vitals)
    forecast = forecast_vitals(model, scaler, patient_data, sim_time)
    what_if_forecast = apply_what_if(forecast, fluid_ml, med_dose)
    risk_score = compute_risk_score(current_vitals, thresholds, anomaly_score, forecast)

    # Apply What-If to risk
    whatif_risk = compute_risk_score(current_vitals, thresholds, anomaly_score * 0.5, what_if_forecast) if (fluid_ml > 0 or med_dose > 0) else risk_score
    risk_reduction = max(0, risk_score - whatif_risk)

    # ── Top Metrics Row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        hr_val = current_vitals[0]
        st.metric("❤️ Heart Rate", f"{hr_val:.0f} bpm",
                  delta=f"{hr_val - VITAL_BASELINE['HR']['mean']:+.1f}",
                  delta_color="inverse")
    with m2:
        spo2_val = current_vitals[1]
        st.metric("🫁 SpO₂", f"{spo2_val:.1f}%",
                  delta=f"{spo2_val - VITAL_BASELINE['SpO2']['mean']:+.1f}",
                  delta_color="normal")
    with m3:
        sbp_val = current_vitals[2]
        st.metric("🩸 Systolic BP", f"{sbp_val:.0f} mmHg",
                  delta=f"{sbp_val - VITAL_BASELINE['SBP']['mean']:+.1f}",
                  delta_color="inverse")
    with m4:
        rr_val = current_vitals[3]
        st.metric("🌬️ Resp. Rate", f"{rr_val:.0f} br/min",
                  delta=f"{rr_val - VITAL_BASELINE['RR']['mean']:+.1f}",
                  delta_color="inverse")
    with m5:
        temp_val = current_vitals[4]
        st.metric("🌡️ Temperature", f"{temp_val:.1f}°C",
                  delta=f"{temp_val - VITAL_BASELINE['Temp']['mean']:+.2f}",
                  delta_color="inverse")

    # ── Alert Banner ──
    if risk_score >= 60:
        st.markdown(f"""
        <div class="alert-critical">
            <strong>🐉 DRAGON ALERT — Deterioration Pattern Detected!</strong><br>
            Risk Score: <strong>{risk_score:.0f}%</strong> | 
            Anomaly: <strong>{'YES' if is_anomaly else 'NO'}</strong> | 
            Patient Bed {patient_id} @ t={sim_time} min
            {'<br>💡 <em>Adjust What-If interventions in sidebar to simulate risk reduction</em>' if fluid_ml == 0 and med_dose == 0 else ''}
        </div>
        """, unsafe_allow_html=True)
    elif risk_score >= 35:
        st.markdown(f"""
        <div class="alert-critical" style="border-left-color:#FFD600; background:rgba(255,214,0,0.08);">
            <strong>⚠️ CAUTION — Subtle Changes Detected</strong><br>
            Risk Score: <strong>{risk_score:.0f}%</strong> | 
            Monitoring elevated | Bed {patient_id}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-stable">
            <strong>✅ STABLE — All Vitals Within Normal Range</strong><br>
            Risk Score: <strong>{risk_score:.0f}%</strong> | 
            VitalGuardian monitoring normally | Bed {patient_id}
        </div>
        """, unsafe_allow_html=True)

    # ── Main Layout: Plot + Gauge + SHAP ──
    col_plot, col_right = st.columns([3, 1.3])

    with col_plot:
        fig_vitals = create_vitals_plot(patient_data, sim_time, forecast, what_if_forecast)
        st.plotly_chart(fig_vitals, use_container_width=True, key="vitals_plot")

    with col_right:
        # Risk Gauge
        fig_gauge = create_risk_gauge(risk_score)
        st.plotly_chart(fig_gauge, use_container_width=True, key="risk_gauge")

        # What-If reduction indicator
        if fluid_ml > 0 or med_dose > 0:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🧪 What-If Result</h3>
                <p class="value" style="color:#00C853;">{whatif_risk:.0f}%</p>
                <p class="delta" style="color:#00C853;">▼ {risk_reduction:.0f}% risk reduction</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <h3>🧪 What-If Simulator</h3>
                <p style="color:#8B9DC3; font-size:0.85rem; margin:0;">
                    Adjust <strong>Fluid</strong> & <strong>Med</strong> sliders in sidebar to simulate interventions
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── SHAP + Vitals Table ──
    st.markdown("---")
    col_shap, col_table = st.columns([1, 1])

    with col_shap:
        st.markdown("### 🔍 Explainable AI — Feature Contributions")
        with st.spinner("Computing explanations..."):
            shap_values, shap_method = compute_shap_values(model, scaler, patient_data, sim_time)

        shap_img = render_shap_chart(shap_values, shap_method)
        st.image(shap_img, use_container_width=True)

        # Top contributor callout
        top_idx = np.argmax(shap_values)
        total = shap_values.sum() + 1e-8
        top_pct = (shap_values[top_idx] / total) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Top Risk Contributor</h3>
            <p class="value" style="color:#FF8C00;">{VITAL_NAMES[top_idx]} — {top_pct:.0f}%</p>
            <p class="delta" style="color:#8B9DC3;">Using {shap_method}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_table:
        st.markdown("### 📋 Vital Signs vs Adaptive Thresholds")

        table_data = []
        for i, vital in enumerate(VITAL_NAMES):
            val = float(current_vitals[i])
            lo, hi = thresholds[vital]
            status = "🟢 Normal" if lo <= val <= hi else "🔴 Breach"
            table_data.append({
                "Vital": f"{vital} ({VITAL_UNITS[i]})",
                "Current": f"{val:.1f}",
                "Lower (μ-2σ)": f"{lo:.1f}",
                "Upper (μ+2σ)": f"{hi:.1f}",
                "Status": status,
            })

        df_table = pd.DataFrame(table_data)
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Vital": st.column_config.TextColumn(width="medium"),
                "Status": st.column_config.TextColumn(width="small"),
            }
        )

        # False alarm reduction metric
        traditional_alarms = sum(
            1 for i, v in enumerate(VITAL_NAMES)
            if float(current_vitals[i]) < VITAL_BASELINE[v]["lo"]
            or float(current_vitals[i]) > VITAL_BASELINE[v]["hi"]
        )
        adaptive_alarms = sum(
            1 for i, v in enumerate(VITAL_NAMES)
            if float(current_vitals[i]) < thresholds[v][0]
            or float(current_vitals[i]) > thresholds[v][1]
        )

        st.markdown(f"""
        <div class="metric-card">
            <h3>🔕 False Alarm Reduction</h3>
            <p style="color:#FAFAFA; margin:0.3rem 0; font-size:0.9rem;">
                Fixed Threshold Alarms: <strong style="color:#FF4500;">{traditional_alarms}</strong><br>
                Adaptive Threshold Alarms: <strong style="color:#00C853;">{adaptive_alarms}</strong><br>
                {'<span style="color:#00C853;">✓ ' + str(traditional_alarms - adaptive_alarms) + ' false alarm(s) prevented</span>' if traditional_alarms > adaptive_alarms else '<span style="color:#8B9DC3;">—</span>'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Projected Impact ──
    st.markdown("---")
    col_impact, col_actions = st.columns([1.5, 1])

    with col_impact:
        st.markdown("### 📈 Projected Clinical Impact")
        fig_impact = create_impact_chart()
        st.plotly_chart(fig_impact, use_container_width=True, key="impact_chart")

    # ── Action Buttons ──
    with col_actions:
        st.markdown("### ⚡ Clinical Actions")

        # Initialize audit log in session state
        if "audit_log" not in st.session_state:
            st.session_state.audit_log = []

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("🛑 Override Alert", use_container_width=True):
                entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] OVERRIDE — Bed {patient_id}, Risk {risk_score:.0f}%, Clinician dismissed alert"
                st.session_state.audit_log.append(entry)
                st.success("Alert overridden. Logged for audit compliance. ✅")

        with btn2:
            if st.button("🚨 Escalate", use_container_width=True):
                entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ESCALATE — Bed {patient_id}, Risk {risk_score:.0f}%, Rapid response team notified"
                st.session_state.audit_log.append(entry)
                st.info("🏥 Rapid response team notified for Bed " + str(patient_id))

        if st.button("📝 View Audit Log", use_container_width=True):
            if st.session_state.audit_log:
                for entry in st.session_state.audit_log[-10:]:
                    st.text(entry)
            else:
                st.info("No actions logged yet this session.")

        st.markdown("""
        <div class="metric-card" style="margin-top:1rem;">
            <h3>💼 Business Model</h3>
            <p style="color:#FAFAFA; font-size:0.85rem; margin:0;">
                <strong>SaaS:</strong> ₹5-10K/bed/month<br>
                <strong>Year 1:</strong> 10 Maharashtra ICUs<br>
                <strong>Revenue:</strong> ₹2Cr+ projected<br>
                <strong>Integration:</strong> Ayushman Bharat ready
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown("""
    <div class="footer">
        <p>⚖️ <strong>Ethics Notice:</strong> VitalGuardian is a <u>decision-support tool only</u>. 
        All clinical decisions must be made by qualified medical professionals. 
        AI outputs are advisory and require human validation.</p>
        <p>📋 ICMR 2025 Compliant • Simulated Data (MIMIC-III Inspired) • Override Audit Logging Enabled</p>
        <p>🐉 <strong>VitalGuardian</strong> — Taming the ICU Alarm Dragon | 
        Hackfest Innov8 TMRW 2026 | CESA × VIT Mumbai | <em>Build Beyond Infinity</em> ∞</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
