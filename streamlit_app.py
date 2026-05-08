import streamlit as st
import pandas as pd
import json
import os
from catboost import CatBoostClassifier


# 1. Page configuration====================================
st.set_page_config(
    page_title="Left-sided IE Risk Prediction",
    page_icon="🫀",
    layout="centered"
)


# 2. Basic paths===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")


# 3. Load schema===========================================
@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


schema = load_schema()

MODEL_FILE = schema.get("model_file", "catboost_train_ro_final.cbm")
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

CUTOFF = float(schema.get("cutoff", 0.5))
FEATURES = schema["features"]


# 4. Load CatBoost model===================================
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


model = load_model()


# 5. Get true feature order from CatBoost model============
MODEL_FEATURE_NAMES = list(model.feature_names_)

if not MODEL_FEATURE_NAMES:
    MODEL_FEATURE_NAMES = [item["model_name"] for item in FEATURES]


SCHEMA_BY_MODEL_NAME = {
    item["model_name"]: item for item in FEATURES
}

missing_features_in_schema = [
    f for f in MODEL_FEATURE_NAMES if f not in SCHEMA_BY_MODEL_NAME
]

if missing_features_in_schema:
    st.error(
        "Some model features are not found in feature_schema.json. "
        "Please check the model_name fields."
    )
    st.write("Missing model features:", missing_features_in_schema)
    st.write("Model expected features:", MODEL_FEATURE_NAMES)
    st.stop()


FEATURES_ORDERED = [
    SCHEMA_BY_MODEL_NAME[f] for f in MODEL_FEATURE_NAMES
]


# 6. Header================================================
st.title("Left-sided Infective Endocarditis Risk Prediction Model")

st.markdown(
    f"""
    This web calculator estimates the probability of **{schema.get("positive_class", "outcomes within 30 days after surgery")}**
    for patients with left-sided infective endocarditis using a CatBoost-based prediction model.

    **Model version:** {schema.get("model_version", "1.0.0")}  
    """
)


# 7. Input form============================================
st.subheader("Input Clinical Variables")

input_values = {}

with st.form("prediction_form"):
    for item in FEATURES_ORDERED:
        api_name = item["api_name"]
        display_name = item["display_name"]
        unit = item.get("unit", "")

        label = f"{display_name}"
        if unit:
            label += f" ({unit})"

        input_values[api_name] = st.number_input(
            label,
            min_value=0.00,
            value=0.00,
            step=0.01,
            format="%.2f"
        )

    submitted = st.form_submit_button("Predict Surgical Risk")


# 8. Prediction============================================
if submitted:
    row = {}

    for model_feature_name in MODEL_FEATURE_NAMES:
        item = SCHEMA_BY_MODEL_NAME[model_feature_name]
        api_name = item["api_name"]

        row[model_feature_name] = float(input_values[api_name])

    df = pd.DataFrame(
        [[row[f] for f in MODEL_FEATURE_NAMES]],
        columns=MODEL_FEATURE_NAMES
    )

    try:
        probability = float(model.predict_proba(df)[0][1])
        probability_percent = probability * 100
        risk_group = "High risk" if probability >= CUTOFF else "Low risk"

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted probability",
                value=f"{probability_percent:.2f}%"
            )

        with col2:
            st.metric(
                label="Risk group",
                value=risk_group
            )

        if risk_group == "High risk":
            st.error(
                f"The predicted probability of 30-day outcomes is "
                f"{probability_percent:.2f}%. This result suggests that the patient "
                f"may have a higher predicted surgical risk."
            )
        else:
            st.success(
                f"The predicted probability of 30-day adverse outcomes is "
                f"{probability_percent:.2f}%. This result suggests that the patient "
                f"may have a lower predicted surgical risk."
            )
        st.warning(
            "This result is intended for research and reference only "
            "and should not replace clinical decision-making."
        )

    except Exception as e:
        st.error(
            "Prediction failed. Please check whether feature names and feature order "
            "match the training data."
        )
        st.exception(e)


# 9. Disclaimer============================================
st.markdown("---")
st.caption(
    schema.get(
        "disclaimer",
        "This model is intended for research and reference only and should not replace clinical decision-making."
    )
)
