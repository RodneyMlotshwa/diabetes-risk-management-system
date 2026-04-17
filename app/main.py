import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import os

# 1. Load Model Assets
base_path = os.path.dirname(__file__)
bundle_path = os.path.join(base_path, 'diabetes_models_bundle.pkl')

try:
    bundle = joblib.load(bundle_path)
    model = bundle["XGBoost"]
    le = bundle["LabelEncoder"]
    explainer = shap.TreeExplainer(model)
    print("✅ Assets Loaded")
except Exception as e:
    print(f"❌ Load Error: {e}")

MODEL_COLUMNS = [
    'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
    'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 
    'family_history_diabetes', 'hypertension_history', 'cardiovascular_history', 
    'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
    'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides', 
    'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c', 
    'diabetes_risk_score', 'diagnosed_diabetes', 'gender_male', 'gender_other', 
    'ethnicity_black', 'ethnicity_hispanic', 'ethnicity_other', 'ethnicity_white', 
    'education_level_highschool', 'education_level_no formal', 
    'education_level_postgraduate', 'income_level_Low', 'income_level_Lower-Middle', 
    'income_level_Middle', 'income_level_Upper-Middle', 'employment_status_Retired', 
    'employment_status_Student', 'employment_status_Unemployed', 
    'smoking_status_Former', 'smoking_status_Never'
]

# 2. App Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.Header(className="custom-header", children=[
        html.H1("BC Analytics", className="title-text"),
        html.P("Precision Diabetes Risk Assessment", className="subtitle-text")
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(className="input-scroll-area", children=[
                html.Div(className="form-section", children=[
                    html.H5("1. Demographics", className="section-title"),
                    dbc.Row([
                        dbc.Col([html.Label("Age"), dbc.Input(id="age-in", type="number", value=45)], width=4),
                        dbc.Col([html.Label("Gender"), dbc.Select(id="gender-in", value="Male", options=[{"label": i, "value": i} for i in ["Male", "Female", "Other"]])], width=4),
                        dbc.Col([html.Label("Ethnicity"), dbc.Select(id="eth-in", value="Asian", options=[{"label": i, "value": i} for i in ["Asian", "White", "Hispanic", "Black", "Other"]])], width=4),
                    ])
                ]),
                html.Div(className="form-section", children=[
                    html.H5("2. Clinical Metrics", className="section-title"),
                    dbc.Row([
                        dbc.Col([html.Label("BMI"), dbc.Input(id="bmi-in", type="number", value=25.0)], width=6),
                        dbc.Col([html.Label("HbA1c (%)"), dbc.Input(id="hba1c-in", type="number", value=5.5)], width=6),
                    ], className="mb-3"),
                    dbc.Row([dbc.Col([html.Label("Fasting Glucose (mg/dL)"), dbc.Input(id="glucose-in", type="number", value=100)], width=12)])
                ]),
                html.Div(className="form-section", children=[
                    html.H5("3. Lifestyle Factors", className="section-title"),
                    dbc.Row([
                        dbc.Col([html.Label("Smoking Status"), dbc.Select(id="smoke-in", value="Never", options=[{"label": i, "value": i} for i in ["Never", "Former", "Current"]])], width=6),
                        dbc.Col([html.Label("Phys. Activity (min/week)"), dbc.Input(id="activity-in", type="number", value=150)], width=6),
                    ])
                ]),
                dbc.Button("Generate Risk Profile", id="run-btn", color="primary", className="custom-btn mt-3")
            ])
        ], width=5),
        dbc.Col([html.Div(id="result-display", className="glass-panel", children=[html.Div("Awaiting Data", className="text-center py-5")])], width=7)
    ])
], fluid=True)

# 3. Prediction Callback
@app.callback(
    Output("result-display", "children"),
    Input("run-btn", "n_clicks"),
    [State("age-in", "value"), State("gender-in", "value"), State("eth-in", "value"), 
     State("smoke-in", "value"), State("activity-in", "value"), State("bmi-in", "value"), 
     State("hba1c-in", "value"), State("glucose-in", "value")],
    prevent_initial_call=True
)
def predict_diabetes(n_clicks, age, gender, eth, smoke, activity, bmi, hba1c, glucose):
    # Map UI inputs to model features
    input_row = {col: 0 for col in MODEL_COLUMNS}
    input_row.update({'Age': age, 'bmi': bmi, 'hba1c': hba1c, 'glucose_fasting': glucose, 'physical_activity_minutes_per_week': activity})

    # Set clinical proxies
    input_row['diagnosed_diabetes'] = 1 if hba1c >= 6.5 or glucose >= 126 else 0
    input_row['glucose_postprandial'] = glucose * 1.4 if glucose else 140
    input_row['diabetes_risk_score'] = (age * 0.1) + (bmi * 0.3) + (hba1c * 1.5)

    # Trigger insulin marker for Type 1 logic
    if hba1c > 9.0 and bmi < 22 and age < 25:
        input_row['insulin_level'] = 0.5
    else:
        input_row['insulin_level'] = 12.0

    # Handle categorical encoding
    if gender == 'Male': input_row['gender_male'] = 1
    elif gender == 'Other': input_row['gender_other'] = 1
    
    eth_key = f'ethnicity_{eth.lower()}'
    if eth_key in input_row: input_row[eth_key] = 1

    if smoke == 'Former': input_row['smoking_status_Former'] = 1
    elif smoke == 'Never': input_row['smoking_status_Never'] = 1

    # Get model prediction
    X_new = pd.DataFrame([input_row])[MODEL_COLUMNS]
    prediction_raw = model.predict(X_new)
    pred_idx = int(prediction_raw[0])
    final_status = le.inverse_transform([pred_idx])[0]
    
    # Apply clinical overrides for consistent results
    if hba1c < 5.7 and glucose < 100:
        final_status = "No Diabetes"
    elif hba1c > 9.0 and bmi < 22 and age < 25:
        final_status = "Type 1"

    # SHAP reasoning with 3D array handling
    shap_v = explainer.shap_values(X_new)
    if isinstance(shap_v, list):
        current_shap_values = shap_v[pred_idx]
    elif len(shap_v.shape) == 3:
        current_shap_values = shap_v[0, pred_idx, :]
    else:
        current_shap_values = shap_v

    # Identify top 3 factors
    vals = np.abs(current_shap_values).flatten()
    top_indices = np.argsort(vals)[-3:][::-1]
    reasons = [MODEL_COLUMNS[i] for i in top_indices]

    # UI Styling logic
    color = "#10b981"
    risk_text = "Low Risk"
    if "Type 1" in final_status or "Type 2" in final_status:
        color = "#ef4444"
        risk_text = "High Clinical Risk"
    elif "Pre" in final_status:
        color = "#f59e0b"
        risk_text = "Elevated Risk"

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number", value=hba1c,
        title={'text': "HbA1c Level %", 'font': {'color': 'white'}},
        gauge={'axis': {'range': [None, 15], 'tickcolor': "white"},
               'bar': {'color': "#3b82f6"},
               'steps': [{'range': [0, 5.7], 'color': "#10b981"},
                         {'range': [5.7, 6.4], 'color': "#f59e0b"},
                         {'range': [6.4, 15], 'color': "#ef4444"}]}
    )).update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=220)

    return html.Div([
        html.Div(className="p-4 rounded-3 mb-4", style={"borderLeft": f"10px solid {color}", "background": "rgba(255,255,255,0.05)"}, children=[
            html.Small("DIAGNOSTIC RESULT", className="text-muted"),
            html.H1(final_status, style={"color": color, "fontWeight": "900", "fontSize": "3.8rem"}),
            dbc.Badge(risk_text, color="light", text_color="dark", className="mt-2")
        ]),
        dcc.Graph(figure=gauge_fig, config={'displayModeBar': False}),
        html.Div(className="mt-3 p-3 bg-dark rounded border border-info", children=[
            html.H6("Model Reasoning", className="text-info small mb-2"),
            html.Ul([html.Li(r.replace('_', ' ').title(), className="small text-light") for r in reasons])
        ])
    ])

if __name__ == "__main__":
   app.run(debug=True, port=8051)