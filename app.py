import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="🌧️ Predict Rain App", layout="centered")
st.title("🌧️ Predict Rain (RainTomorrow)")

# Load models and encoders
@st.cache_resource
def load_models():
    return {
        "scaler": joblib.load("saved_models/scaler.joblib"),
        "pca": joblib.load("saved_models/pca_transformer.joblib"),
        "rf": joblib.load("saved_models/random_forest_classifier_pca.joblib"),
        "dt": joblib.load("saved_models/decision_tree_classifier_pca.joblib"),
        "encoders": joblib.load("saved_models/label_encoders.joblib"),
        "acc_rf": joblib.load("saved_models/accuracy_rf.joblib"),
        "acc_dt": joblib.load("saved_models/accuracy_dt.joblib"),
    }

models = load_models()
scaler = models["scaler"]
pca = models["pca"]
rf_model = models["rf"]
dt_model = models["dt"]
label_encoders = models["encoders"]
accuracy_rf = models["acc_rf"]
accuracy_dt = models["acc_dt"]

# Default values for all possible inputs
default_values_full = {
    "Location": "Sydney",
    "MinTemp": 12.0,
    "MaxTemp": 23.0,
    "Rainfall": 0.0,
    "Evaporation": 3.2,
    "Sunshine": 9.8,
    "WindGustDir": "NW",
    "WindGustSpeed": 39.0,
    "WindDir9am": "WNW",
    "WindDir3pm": "WNW",
    "WindSpeed9am": 13.0,
    "WindSpeed3pm": 19.0,
    "Humidity9am": 70.0,
    "Humidity3pm": 52.0,
    "Pressure9am": 1010.0,
    "Pressure3pm": 1010.0,
    "Cloud9am": 5,
    "Cloud3pm": 5,
    "Temp9am": 20.0,
    "Temp3pm": 22.0,
    "RainToday": 0
}

# Chọn 10 features đúng thứ tự
selected_features = [
    "Humidity3pm", "RainToday", "Cloud3pm", "Humidity9am", "Cloud9am",
    "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "MinTemp"
]

# === CHỌN CÁCH NHẬP DỮ LIỆU ===
input_mode = st.radio("📅 How do you want to input data?", [
    "Manual input", "Upload CSV file"
])
def fill_missing_with_defaults(df):
    for col, default in default_values_full.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
    return df

if input_mode == "Manual input":
    def parse_float(value):
        try:
            return float(value)
        except ValueError:
            return None
    try:
        with st.form("input_form"):
            st.subheader("🔢 Input weather forecast data:")
            location = st.text_input("Location", "")
            min_temp = parse_float(st.text_input("MinTemp (°C)", ""))
            max_temp = parse_float(st.text_input("MaxTemp (°C)", ""))
            rainfall = parse_float(st.text_input("Rainfall (mm)", ""))
            evaporation = parse_float(st.text_input("Evaporation (mm)", ""))
            sunshine = parse_float(st.text_input("Sunshine (hours)", ""))
            wind_gust_dir = st.text_input("WindGustDir", "NW")
            wind_gust_speed = parse_float(st.text_input("WindGustSpeed", ""))
            wind_dir_9am = st.text_input("WindDir9am", "WNW")
            wind_dir_3pm = st.text_input("WindDir3pm", "WNW")
            wind_speed_9am = parse_float(st.text_input("WindSpeed9am", ""))
            wind_speed_3pm = parse_float(st.text_input("WindSpeed3pm", ""))
            humidity_9am = parse_float(st.text_input("Humidity9am (%)", ""))
            humidity_3pm = parse_float(st.text_input("Humidity3pm (%)", ""))
            pressure_9am = parse_float(st.text_input("Pressure9am (hPa)", ""))
            pressure_3pm = parse_float(st.text_input("Pressure3pm (hPa)", ""))
            cloud_9am = parse_float(st.text_input("Cloud9am (0-8)", ""))
            cloud_3pm = parse_float(st.text_input("Cloud3pm (0-8)", ""))
            temp_9am = parse_float(st.text_input("Temp9am (°C)", ""))
            temp_3pm = parse_float(st.text_input("Temp3pm (°C)", ""))
            rain_today = st.selectbox("RainToday", ["", "Yes", "No"])
            model_type = st.selectbox("🧠Select a model", ["Random Forest", "Decision Tree"])
            submit = st.form_submit_button("Predict")

        if submit:
            full_input = pd.DataFrame([{k: v for k, v in {
                "Location": location,
                "MinTemp": min_temp,
                "MaxTemp": max_temp,
                "Rainfall": rainfall,
                "Evaporation": evaporation,
                "Sunshine": sunshine,
                "WindGustDir": wind_gust_dir,
                "WindGustSpeed": wind_gust_speed,
                "WindDir9am": wind_dir_9am,
                "WindDir3pm": wind_dir_3pm,
                "WindSpeed9am": wind_speed_9am,
                "WindSpeed3pm": wind_speed_3pm,
                "Humidity9am": humidity_9am,
                "Humidity3pm": humidity_3pm,
                "Pressure9am": pressure_9am,
                "Pressure3pm": pressure_3pm,
                "Cloud9am": cloud_9am,
                "Cloud3pm": cloud_3pm,
                "Temp9am": temp_9am,
                "Temp3pm": temp_3pm,
                "RainToday": rain_today
            }.items()}])

            full_input = fill_missing_with_defaults(full_input)
            input_df = full_input[selected_features].copy()
            if 'RainToday' in input_df.columns:
                input_df['RainToday'] = input_df['RainToday'].map({'Yes': 1, 'No': 0})

            # Chỉ encode những cột object khác (nếu có)
            for col in input_df.columns:
                if col in label_encoders and input_df[col].dtype == object:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

            X_scaled = scaler.transform(input_df)
            X_pca = pca.transform(X_scaled)

            model = rf_model if model_type == "Random Forest" else dt_model
            accuracy = accuracy_rf if model_type == "Random Forest" else accuracy_dt
            prediction = model.predict(X_pca)[0]
            result_label = {0: "No", 1: "Yes"}.get(prediction, str(prediction))
            emoji = "☔" if prediction == 1 else "🌤️"

            st.success(f"🌟 Weather prediction result: **{emoji} {result_label}** (by {model_type})")
            st.info(f"📊 Model accuracy on test data: **{accuracy*100:.2f}%**")
            proba = model.predict_proba(X_pca)[0]
            st.subheader("🧪 Probability of RainTomorrow:")
            st.bar_chart({"No": proba[0], "Yes": proba[1]})
    except Exception as e:
        st.error(f"❌ Error fetching weather: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("📁 Upload a CSV file with input data", type=["csv"])
    model_type = st.selectbox("🧠Select a model", ["Random Forest", "Decision Tree"])

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df = fill_missing_with_defaults(uploaded_df)

        try:
            input_df = uploaded_df[selected_features].copy()
        except KeyError as e:
            st.error(f"❌ Missing required columns: {e}")
        else:
           if 'RainToday' in input_df.columns:
            input_df['RainToday'] = input_df['RainToday'].map({'Yes': 1, 'No': 0})

            # Chỉ encode những cột object khác (nếu có)
            for col in input_df.columns:
                if col in label_encoders and input_df[col].dtype == object:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

            X_scaled = scaler.transform(input_df)
            X_pca = pca.transform(X_scaled)

            model = rf_model if model_type == "Random Forest" else dt_model
            accuracy = accuracy_rf if model_type == "Random Forest" else accuracy_dt
            predictions = model.predict(X_pca)

            output_df = uploaded_df.copy()
            output_df["Prediction"] = predictions
            output_df["PredictionLabel"] = output_df["Prediction"].map({0: "No", 1: "Yes"})

            csv = output_df.to_csv(index=False).encode("utf-8")
            st.success("✅ Prediction completed. Download below:")
            st.download_button(
                label="📅 Download prediction result as CSV",
                data=csv,
                file_name="rain_prediction_result.csv",
                mime="text/csv"
            )