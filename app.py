from pathlib import Path
import pickle

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "customer_spend_model.pkl"


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run `python train_model.py` first."
        )
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


st.set_page_config(
    page_title="Customer Spend Predictor",
    page_icon="C",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17211b;
        --muted: #65736a;
        --paper: #f5f7f1;
        --mint: #dceee1;
        --green: #1e6b4a;
        --coral: #ef765e;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { visibility: hidden; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    h1 { font-size: clamp(2.4rem, 5vw, 4.8rem); line-height: .95; letter-spacing: 0; }
    [data-testid="stNumberInput"] label, [data-testid="stNumberInput"] label p { color: var(--ink) !important; }
    [data-testid="stNumberInput"] [data-testid="stWidgetLabelHelp"] { color: var(--muted) !important; }
    .hero { padding: 3rem 0 1.5rem; max-width: 820px; }
    .kicker { color: var(--green); font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero-copy { color: var(--muted); font-size: 1.08rem; max-width: 590px; }
    .panel { background: white; border: 1px solid #e2e9df; border-radius: 8px; padding: 1.45rem 1.6rem; box-shadow: 0 14px 35px rgba(23, 33, 27, .06); }
    .panel-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; margin-bottom: .2rem; }
    .panel-note { color: var(--muted); font-size: .88rem; margin-bottom: 1.1rem; }
    .stat { border-left: 3px solid var(--coral); padding-left: .85rem; }
    .stat-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .stat-value { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; }
    .result { background: var(--green); border-radius: 8px; padding: 1.6rem; color: white; margin-top: 1.2rem; }
    .result-label { color: #bfe0ca; font-size: .8rem; text-transform: uppercase; letter-spacing: .1em; }
    .result-value { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3.3rem); font-weight: 700; margin-top: .2rem; }
    .result-foot { color: #dceee1; font-size: .85rem; margin-top: .7rem; }
    div.stButton > button, div[data-testid="stFormSubmitButton"] button { background: var(--coral); color: white; border: 0; border-radius: 5px; font-weight: 700; min-height: 2.8rem; }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { background: #d95f49; color: white; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="kicker">Ecommerce intelligence / 01</div>
        <h1>Know the value<br>of every visit.</h1>
        <p class="hero-copy">Turn customer engagement signals into a clear yearly spend estimate.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    artifact = load_artifact()
except (FileNotFoundError, pickle.UnpicklingError, KeyError) as error:
    st.error(str(error))
    st.stop()

form_column, insight_column = st.columns([1.35, 1], gap="large")

with form_column:
    st.markdown(
        '<div class="panel"><div class="panel-title">Customer signals</div><div class="panel-note">Enter the engagement profile to score.</div>',
        unsafe_allow_html=True,
    )
    with st.form("prediction_form"):
        input_left, input_right = st.columns(2, gap="medium")
        with input_left:
            avg_session_length = st.number_input(
                "Average session length (minutes)",
                min_value=0.0,
                value=33.0,
                step=0.1,
                help="Average session length in minutes.",
            )
            time_on_website = st.number_input(
                "Time on website (minutes/day)",
                min_value=0.0,
                value=37.0,
                step=0.1,
                help="Average daily time spent on the website.",
            )
        with input_right:
            time_on_app = st.number_input(
                "Time on app (minutes/day)",
                min_value=0.0,
                value=12.0,
                step=0.1,
                help="Average daily time spent using the mobile app.",
            )
            length_of_membership = st.number_input(
                "Length of membership (years)",
                min_value=0.0,
                value=3.5,
                step=0.1,
                help="Membership duration in years.",
            )
        submitted = st.form_submit_button("Predict yearly spend", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with insight_column:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Model snapshot</div>
            <div class="panel-note">The estimate is based on the included customer dataset.</div>
            <div class="stat"><div class="stat-label">Method</div><div class="stat-value">Linear regression</div></div><br>
            <div class="stat"><div class="stat-label">Signals used</div><div class="stat-value">4 engagement features</div></div><br>
            <div class="stat"><div class="stat-label">Output</div><div class="stat-value">Yearly amount spent</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if submitted:
    values = [[
        avg_session_length,
        time_on_app,
        time_on_website,
        length_of_membership,
    ]]
    inputs = pd.DataFrame(values, columns=artifact["features"])
    prediction = float(artifact["model"].predict(inputs)[0])
    monthly_equivalent = prediction / 12
    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">Estimated yearly amount spent</div>
            <div class="result-value">${prediction:,.2f}</div>
            <div class="result-foot">Monthly equivalent: ${monthly_equivalent:,.2f}. The model predicts yearly spending.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )