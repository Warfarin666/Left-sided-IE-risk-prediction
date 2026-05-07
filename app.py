from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from catboost import CatBoostClassifier
import pandas as pd
import json
import os

# 1. Basic paths===========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(BASE_DIR, "feature_schema.json")


# 2. Load feature schema===================================
if not os.path.exists(SCHEMA_PATH):
    raise FileNotFoundError("feature_schema.json not found.")

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)


MODEL_FILE = schema.get("model_file", "catboost_model.cbm")
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILE)

CUTOFF = float(schema.get("cutoff", 0.5))
FEATURES = schema["features"]

API_FEATURE_NAMES = [item["api_name"] for item in FEATURES]
MODEL_FEATURE_NAMES = [item["model_name"] for item in FEATURES]


# 3. Load CatBoost model===================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_FILE} not found.")

model = CatBoostClassifier()
model.load_model(MODEL_PATH)


# 4. Initialize FastAPI====================================
app = FastAPI(
    title=schema.get("model_name", "CatBoost Prediction Model"),
    version=schema.get("model_version", "1.0.0"),
    description="Online prediction API for a CatBoost-based medical risk model."
)


# 5. Request format========================================
class PredictRequest(BaseModel):
    features: dict


# 6. Home page with a simple web form======================
@app.get("/", response_class=HTMLResponse)
def home():
    feature_inputs = ""

    for item in FEATURES:
        api_name = item["api_name"]
        display_name = item["display_name"]
        unit = item.get("unit", "")

        feature_inputs += f"""
        <label>{display_name} ({unit})</label>
        <input type="number" step="any" id="{api_name}" placeholder="Enter {display_name}" required>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{schema.get("model_name")}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 760px;
                margin: 40px auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
            h1 {{
                color: #1f2937;
                font-size: 26px;
                margin-bottom: 8px;
            }}
            .subtitle {{
                color: #6b7280;
                margin-bottom: 24px;
            }}
            label {{
                display: block;
                margin-top: 14px;
                font-weight: bold;
                color: #374151;
            }}
            input {{
                width: 100%;
                padding: 10px;
                margin-top: 6px;
                border: 1px solid #d1d5db;
                border-radius: 6px;
                font-size: 15px;
                box-sizing: border-box;
            }}
            button {{
                margin-top: 24px;
                width: 100%;
                padding: 12px;
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #1d4ed8;
            }}
            .result {{
                margin-top: 24px;
                padding: 18px;
                border-radius: 8px;
                background-color: #f3f4f6;
                display: none;
            }}
            .high {{
                color: #b91c1c;
                font-weight: bold;
            }}
            .low {{
                color: #047857;
                font-weight: bold;
            }}
            .note {{
                margin-top: 18px;
                color: #6b7280;
                font-size: 13px;
                line-height: 1.5;
            }}
            .links {{
                margin-top: 16px;
                font-size: 14px;
            }}
            .links a {{
                color: #2563eb;
                text-decoration: none;
                margin-right: 12px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{schema.get("model_name")}</h1>
            <div class="subtitle">
                Model version: {schema.get("model_version")} |
                Cutoff: {CUTOFF}
            </div>

            {feature_inputs}

            <button onclick="submitPrediction()">Predict Risk</button>

            <div id="result" class="result"></div>

            <div class="links">
                <a href="/docs" target="_blank">API Documentation</a>
                <a href="/features" target="_blank">Feature Information</a>
            </div>

            <div class="note">
                {schema.get("disclaimer")}
            </div>
        </div>

        <script>
            async function submitPrediction() {{
                const featureNames = {API_FEATURE_NAMES};
                let features = {{}};

                for (const name of featureNames) {{
                    const value = document.getElementById(name).value;
                    if (value === "") {{
                        alert("Please complete all required fields.");
                        return;
                    }}
                    features[name] = parseFloat(value);
                }}

                const response = await fetch("/predict", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{features: features}})
                }});

                const resultDiv = document.getElementById("result");
                resultDiv.style.display = "block";

                const data = await response.json();

                if (!response.ok) {{
                    resultDiv.innerHTML = "<strong>Error:</strong><br>" + JSON.stringify(data, null, 2);
                    return;
                }}

                const riskClass = data.risk_group === "High risk" ? "high" : "low";

                resultDiv.innerHTML = `
                    <h3>Prediction Result</h3>
                    <p><strong>Predicted probability:</strong> ${{data.predicted_probability_percent}}%</p>
                    <p><strong>Cutoff:</strong> ${{data.cutoff}}</p>
                    <p><strong>Risk group:</strong> <span class="${{riskClass}}">${{data.risk_group}}</span></p>
                    <p><strong>Positive class:</strong> ${{data.positive_class}}</p>
                `;
            }}
        </script>
    </body>
    </html>
    """

    return html_content


# 7. Health check==========================================
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "The prediction service is running."
    }


# 8. Show feature information==============================
@app.get("/features")
def get_features():
    return {
        "model_name": schema.get("model_name"),
        "model_abbreviation": schema.get("model_abbreviation"),
        "model_version": schema.get("model_version"),
        "cutoff": CUTOFF,
        "features": FEATURES
    }


# 9. Prediction endpoint===================================
@app.post("/predict")
def predict(request: PredictRequest):
    input_data = request.features

    missing_features = [f for f in API_FEATURE_NAMES if f not in input_data]

    if missing_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features
            }
        )

    row = {}
    invalid_features = []

    for item in FEATURES:
        api_name = item["api_name"]
        model_name = item["model_name"]
        value = input_data.get(api_name)

        try:
            row[model_name] = float(value)
        except Exception:
            invalid_features.append(api_name)

    if invalid_features:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Some features are not numeric.",
                "invalid_features": invalid_features
            }
        )

    df = pd.DataFrame([row], columns=MODEL_FEATURE_NAMES)

    try:
        probability = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Prediction failed. Please check whether feature names match the training data.",
                "error": str(e),
                "expected_model_features": MODEL_FEATURE_NAMES
            }
        )

    risk_group = "High risk" if probability >= CUTOFF else "Low risk"

    return {
        "model_name": schema.get("model_name"),
        "model_abbreviation": schema.get("model_abbreviation"),
        "model_version": schema.get("model_version"),
        "predicted_probability": round(probability, 4),
        "predicted_probability_percent": round(probability * 100, 2),
        "cutoff": CUTOFF,
        "risk_group": risk_group,
        "positive_class": schema.get("positive_class"),
        "disclaimer": schema.get("disclaimer")
    }
