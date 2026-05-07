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

API_FEATURE_NAMES = [item["api_name"] for item in FEATURES]
MODEL_FEATURE_NAMES = [item["model_name"] for item in FEATURES]


# 4. Load CatBoost model===================================
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


model = load_model()


# 5. Header================================================
st.title("Left-sided Infective Endocarditis Risk Prediction Model")

st.markdown(
    f"""
    This web calculator estimates the probability of **{schema.get("positive_class", "30-day adverse outcome")}**
    using a CatBoost-based prediction model.

    **Model version:** {schema.get("model_version", "1.0.0")}  
    **Cutoff:** {CUTOFF}
    """
)


# 6. Input form============================================
st.subheader("Input Clinical Variables")

input_values = {}

with st.form("prediction_form"):
    for item in FEATURES:
        api_name = item["api_name"]
        display_name = item["display_name"]
        unit = item.get("unit", "")
        
        label = f"{display_name}"
        if unit:
            label += f" ({unit})"

        input_values[api_name] = st.number_input(
            label,
            value=0.0,
            step=0.1,
            format="%.4f"
        )

    submitted = st.form_submit_button("Predict Risk")


# 7. Prediction============================================
if submitted:
    row = {}

    for item in FEATURES:
        api_name = item["api_name"]
        model_name = item["model_name"]
        row[model_name] = float(input_values[api_name])

    df = pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)

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
                f"The predicted probability is {probability_percent:.2f}%, "
                f"which is greater than or equal to the cutoff of {CUTOFF}."
            )
        else:
            st.success(
                f"The predicted probability is {probability_percent:.2f}%, "
                f"which is lower than the cutoff of {CUTOFF}."
            )

        with st.expander("Show model input details"):
            st.dataframe(df)

    except Exception as e:
        st.error("Prediction failed. Please check whether feature names match the training data.")
        st.exception(e)


# 8. Feature information===================================
with st.expander("Required predictors"):
    feature_table = pd.DataFrame([
        {
            "API variable": item["api_name"],
            "Model variable": item["model_name"],
            "Clinical variable": item["display_name"],
            "Unit": item.get("unit", "")
        }
        for item in FEATURES
    ])
    st.dataframe(feature_table, use_container_width=True)


# 9. Disclaimer============================================
st.markdown("---")
st.caption(
    schema.get(
        "disclaimer",
        "This model is intended for research and clinical decision-support purposes only and should not replace clinical judgment."
    )
)
