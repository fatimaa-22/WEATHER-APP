import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Energy Prediction App",
    page_icon="⚡",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
with open("energy_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>
    ⚡ Energy Consumption Prediction System
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Predict application energy usage based on weather conditions
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("🌦 Weather Inputs")

Pressure = st.sidebar.number_input("Pressure (hPa)", value=1012.0)
global_radiation = st.sidebar.number_input("Global Radiation", value=350.0)
temp_mean = st.sidebar.number_input("Mean Temperature (°C)", value=28.0)
temp_min = st.sidebar.number_input("Min Temperature (°C)", value=24.0)
temp_max = st.sidebar.number_input("Max Temperature (°C)", value=32.0)
Wind_Speed = st.sidebar.number_input("Wind Speed (km/h)", value=5.0)
Wind_Bearing = st.sidebar.number_input("Wind Bearing (°)", value=180.0)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Input Weather Summary")

    input_df = pd.DataFrame({
        "Parameter": [
            "Pressure", "Global Radiation", "Temp Mean",
            "Temp Min", "Temp Max", "Wind Speed", "Wind Bearing"
        ],
        "Value": [
            Pressure, global_radiation, temp_mean,
            temp_min, temp_max, Wind_Speed, Wind_Bearing
        ]
    })

    st.table(input_df)

with col2:
    st.subheader("🔮 Energy Prediction")

    if st.button("⚡ Predict Energy Consumption", use_container_width=True):
        input_data = np.array([[
            Pressure,
            global_radiation,
            temp_mean,
            temp_min,
            temp_max,
            Wind_Speed,
            Wind_Bearing
        ]])

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"""
            <div style="
                background-color:#e8f4ff;
                padding:30px;
                border-radius:15px;
                text-align:center;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            ">
                <h2 style="color:#1f77b4;">Predicted Energy</h2>
                <h1 style="color:#ff7f0e;">{prediction:.4f}</h1>
                <p>(Normalized Energy Consumption)</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- VISUALIZATION ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("📈 Weather Impact Visualization")

chart_df = pd.DataFrame({
    "Feature": [
        "Pressure", "Global Radiation", "Temp Mean",
        "Temp Min", "Temp Max", "Wind Speed", "Wind Bearing"
    ],
    "Value": [
        Pressure, global_radiation, temp_mean,
        temp_min, temp_max, Wind_Speed, Wind_Bearing
    ]
})

st.bar_chart(chart_df.set_index("Feature"))

# ---------------- FOOTER ----------------
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
    Final Year Project | Energy Prediction Using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)
