import streamlit as st
import pandas as pd
import json
import os
from catboost import CatBoostClassifier


# =========================================================
# 1. Page configuration
# =========================================================
st.set_page_config(
    page_title="Left-sided IE Risk Prediction",
    page_icon="🫀",
    layout="centered"
)


# =========================================================
# 2. Basic paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")


# =========================================================
# 3. Load schema
# =========================================================
@st.cache_data
def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


schema = load_schema()

MODEL_FILE = schema.get("model_file", "catboost_train_ro_final.cbm")
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

CUTOFF = float(schema.get("cutoff", 0.5))
FEATURES = schema["features"]


# =========================================================
# 4. Load CatBoost model
# =========================================================
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return model


model = load_model()


# =========================================================
# 5. Get true feature order from CatBoost model
# =========================================================
MODEL_FEATURE_NAMES = list(model.feature_names_)

if not MODEL_FEATURE_NAMES:
    MODEL_FEATURE_NAMES = [item["model_name"] for item in FEATURES]


# Build mapping from model feature name to schema item
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


# Reorder features according to the model's internal feature order
FEATURES_ORDERED = [
    SCHEMA_BY_MODEL_NAME[f] for f in MODEL_FEATURE_NAMES
]


# =========================================================
# 6. Header
# =========================================================
st.title("Left-sided Infective Endocarditis Risk Prediction Model")

st.markdown(
    f"""
    This web calculator estimates the probability of **{schema.get("positive_class", "30-day adverse outcome")}**
    using a CatBoost-based prediction model.

    **Model version:** {schema.get("model_version", "1.0.0")}  
    **Cutoff:** {CUTOFF}
    """
)


# =========================================================
# 7. Input form
# =========================================================
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
            value=0.0,
            step=0.1,
            format="%.4f"
        )

    submitted = st.form_submit_button("Predict Risk")


# =========================================================
# 8. Prediction
# =========================================================
if submitted:
    row = {}

    for model_feature_name in MODEL_FEATURE_NAMES:
        item = SCHEMA_BY_MODEL_NAME[model_feature_name]
        api_name = item["api_name"]

        row[model_feature_name] = float(input_values[api_name])

    # Important: dataframe columns must follow the model's internal feature order
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
                f"The predicted probability is {probability_percent:.2f}%, "
                f"which is greater than or equal to the cutoff of {CUTOFF}."
            )
        else:
            st.success(
                f"The predicted probability is {probability_percent:.2f}%, "
                f"which is lower than the cutoff of {CUTOFF}."
            )

        with st.expander("Show model input details"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Please check whether feature names and feature order match the training data.")
        st.exception(e)


# =========================================================
# 9. Feature information
# =========================================================
with st.expander("Required predictors and model feature order"):
    feature_table = pd.DataFrame([
        {
            "Order in model": i + 1,
            "API variable": item["api_name"],
            "Model variable": item["model_name"],
            "Clinical variable": item["display_name"],
            "Unit": item.get("unit", "")
        }
        for i, item in enumerate(FEATURES_ORDERED)
    ])

    st.dataframe(feature_table, use_container_width=True)


# =========================================================
# 10. Model internal feature names
# =========================================================
with st.expander("Model internal feature names"):
    st.write(MODEL_FEATURE_NAMES)


# =========================================================
# 11. Disclaimer
# =========================================================
st.markdown("---")
st.caption(
    schema.get(
        "disclaimer",
        "This model is intended for research and clinical decision-support purposes only and should not replace clinical judgment."
    )
)
