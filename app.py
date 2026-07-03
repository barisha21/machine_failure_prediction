import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = r"C:\Users\Admin\Documents\machine_failure\ipynb\machine_failure_prediction_model.pkl"
SCALER_PATH = r"C:\Users\Admin\Documents\machine_failure\ipynb\scaler.pkl"
ENCODER_PATH = r"C:\Users\Admin\Documents\machine_failure\ipynb\label_encoder.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Background */

.stApp{
background:#F8F9FC;
}

/* Hide Streamlit */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#1E293B;
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Main Title */

.main-title{
font-size:48px;
font-weight:800;
text-align:center;
color:#1E3A8A;
margin-bottom:10px;
}

/* Subtitle */

.sub-title{
font-size:22px;
text-align:center;
color:#64748B;
margin-bottom:25px;
}

/* Overview Card */

.overview{
background:white;
padding:30px;
border-radius:18px;
box-shadow:0px 6px 20px rgba(0,0,0,.08);
border-left:8px solid #2563EB;
}

/* Input Card */

.input-card{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 6px 20px rgba(0,0,0,.08);
}

/* Result Card */

.result-card{
background:#ECFDF5;
padding:25px;
border-radius:18px;
box-shadow:0px 5px 15px rgba(0,0,0,.08);
border-left:8px solid #16A34A;
}

/* Failure Card */

.failure-card{
background:#FEF2F2;
padding:25px;
border-radius:18px;
box-shadow:0px 5px 15px rgba(0,0,0,.08);
border-left:8px solid #DC2626;
}

/* Dashboard Card */

.dashboard-card{
background:white;
padding:20px;
border-radius:15px;
text-align:center;
box-shadow:0px 5px 15px rgba(0,0,0,.08);
}

/* Predict Button */

.stButton>button{

width:100%;

height:60px;

border:none;

border-radius:12px;

font-size:22px;

font-weight:bold;

color:white;

background:linear-gradient(90deg,#2563EB,#3B82F6);

}

.stButton>button:hover{

background:linear-gradient(90deg,#1D4ED8,#2563EB);

}

/* Footer */

.footer{

text-align:center;

font-size:16px;

color:#64748B;

margin-top:30px;

}

</style>
""", unsafe_allow_html=True)
# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/factory.png",
        width=80
    )

    st.markdown("## 🏭 Machine Failure Prediction")

    st.markdown("---")

    st.success("### 🤖 Machine Learning Model")
    st.write("Random Forest Classifier")

    st.markdown("---")

    st.info("""
### 💻 Technologies

- Python
- Streamlit
- Scikit-Learn
- NumPy
- Pandas
- Pickle
""")

    st.markdown("---")

    st.metric(
        label="🎯 Model Accuracy",
        value="98.40%"
    )

    st.markdown("---")

    st.markdown("### 📂 Dataset")

    st.write("AI4I 2020 Predictive Maintenance")

    st.markdown("---")

    st.caption("Developed for M.Sc Computer Science Project")


# =====================================================
# MAIN TITLE
# =====================================================

st.markdown("""
<div class="main-title">

🏭 Machine Failure Prediction System

</div>

<div class="sub-title">

AI Powered Predictive Maintenance using Machine Learning

</div>
""", unsafe_allow_html=True)


# =====================================================
# PROJECT OVERVIEW
# =====================================================

st.markdown("""

<div class="overview">

<h2 style="color:#1E3A8A;">

📖 Project Overview

</h2>

<p style="font-size:18px; color:#555; line-height:1.8;">

This application predicts whether an industrial machine
is likely to fail based on its operating parameters.

The prediction is performed using a trained
<b>Random Forest Machine Learning Model</b>
developed with the
<b>AI4I 2020 Predictive Maintenance Dataset.</b>

The system assists industries in reducing downtime,
improving maintenance planning,
and increasing machine reliability.

</p>

</div>

""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =====================================================
# QUICK INFORMATION
# =====================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.info("🏭 Machine Monitoring")

with c2:
    st.info("🤖 AI Prediction")

with c3:
    st.info("⚡ Real-Time Analysis")

with c4:
    st.info("📊 Predictive Maintenance")

st.markdown("<br>", unsafe_allow_html=True)
# =====================================================
# MACHINE INPUT SECTION
# =====================================================

st.markdown("""
<h2 style='text-align:center;color:#1E3A8A;'>
📝 Enter Machine Details
</h2>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.container():

    left, right = st.columns(2)

    # ==========================
    # LEFT COLUMN
    # ==========================

    with left:

        st.markdown("### 🏭 Machine Information")

        machine_type = st.selectbox(
            "Machine Type",
            ["Select", "L", "M", "H"]
        )

        rotational_speed = st.number_input(
            "⚙ Rotational Speed (RPM)",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=1
        )

        tool_wear = st.number_input(
            "⏱ Tool Wear (min)",
            min_value=0,
            max_value=300,
            value=10,
            step=1
        )

    # ==========================
    # RIGHT COLUMN
    # ==========================

    with right:

        st.markdown("### 🌡 Operating Conditions")

        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=298.0,
            step=0.1
        )

        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=308.0,
            step=0.1
        )

        torque = st.number_input(
            "🔩 Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# INPUT SUMMARY
# =====================================================

st.markdown("### 📋 Current Input")

s1, s2, s3 = st.columns(3)

with s1:
    st.metric("Machine", machine_type)

with s2:
    st.metric("Air Temp", f"{air_temp:.1f} K")

with s3:
    st.metric("Process Temp", f"{process_temp:.1f} K")

s4, s5, s6 = st.columns(3)

with s4:
    st.metric("RPM", rotational_speed)

with s5:
    st.metric("Torque", f"{torque:.1f} Nm")

with s6:
    st.metric("Tool Wear", f"{tool_wear} min")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICT BUTTON
# =====================================================

predict = st.button(
    "🚀 Predict Machine Failure",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    # Validate Machine Type
    if machine_type == "Select":

        st.warning("⚠ Please select a Machine Type.")
        st.stop()

    # Encode Machine Type
    machine_encoded = encoder.transform([machine_type])[0]

    # Create Input Array
    input_data = np.array([[
        machine_encoded,
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]])

    # Scale Features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # RESULT CARD
    # =====================================================

    if prediction == 0:

        st.markdown("""
        <div class="result-card">

        <h1 style="text-align:center;color:#16A34A;">
        ✅ Machine is Healthy
        </h1>

        <hr>

        <p style="font-size:18px;text-align:center;">

        No machine failure is predicted.

        The machine is operating normally.

        Preventive maintenance is not immediately required.

        </p>

        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown("""
        <div class="failure-card">

        <h1 style="text-align:center;color:#DC2626;">
        ⚠ Machine Failure Predicted
        </h1>

        <hr>

        <p style="font-size:18px;text-align:center;">

        The model predicts a possible machine failure.

        Preventive maintenance is recommended.

        Inspect the machine before continued operation.

        </p>

        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # MACHINE HEALTH STATUS
    # =====================================================

    st.subheader("🏥 Machine Health Status")

    if prediction == 0:

        st.progress(100)

        st.success("Machine Health : Excellent")

    else:

        st.progress(35)

        st.error("Machine Health : Poor")
        # =====================================================
# MACHINE HEALTH DASHBOARD
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h2 style='text-align:center;color:#1E3A8A;'>
📊 Machine Health Dashboard
</h2>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

# --------------------------------
# Machine Type
# --------------------------------

with c1:

    st.markdown(f"""
    <div class='dashboard-card'>

    <h1>🏭</h1>

    <h4>Machine Type</h4>

    <h2 style='color:#2563EB;'>{machine_type}</h2>

    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# Air Temperature
# --------------------------------

with c2:

    st.markdown(f"""
    <div class='dashboard-card'>

    <h1>🌡</h1>

    <h4>Air Temperature</h4>

    <h2 style='color:#EA580C;'>{air_temp:.1f} K</h2>

    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# Rotational Speed
# --------------------------------

with c3:

    st.markdown(f"""
    <div class='dashboard-card'>

    <h1>⚙</h1>

    <h4>Speed</h4>

    <h2 style='color:#16A34A;'>{rotational_speed}</h2>

    <p>RPM</p>

    </div>
    """, unsafe_allow_html=True)

# --------------------------------
# Torque
# --------------------------------

with c4:

    st.markdown(f"""
    <div class='dashboard-card'>

    <h1>🔩</h1>

    <h4>Torque</h4>

    <h2 style='color:#DC2626;'>{torque:.1f}</h2>

    <p>Nm</p>

    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# =====================================================
# PARAMETER ANALYSIS
# =====================================================

st.markdown("""
<h2 style='text-align:center;color:#1E3A8A;'>
📈 Machine Parameter Analysis
</h2>
""", unsafe_allow_html=True)

chart_df = pd.DataFrame({
    "Parameter": [
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear"
    ],
    "Value": [
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]
})

st.bar_chart(
    chart_df,
    x="Parameter",
    y="Value",
    use_container_width=True
)

st.markdown("<br>", unsafe_allow_html=True)
# =====================================================
# PARAMETER SUMMARY
# =====================================================

st.markdown("### 📋 Parameter Summary")

summary = pd.DataFrame({
    "Parameter": [
        "Machine Type",
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],
    "Value": [
        machine_type,
        f"{air_temp:.1f} K",
        f"{process_temp:.1f} K",
        f"{rotational_speed} RPM",
        f"{torque:.1f} Nm",
        f"{tool_wear} min"
    ]
})

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True
)