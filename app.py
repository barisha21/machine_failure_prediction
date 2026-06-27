import streamlit as st
import pickle
import numpy as np


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

with open("C:\\Users\\Admin\\Documents\\machine_failure\\ipynb\\machine_failure_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:\\Users\\Admin\\Documents\\machine_failure\\ipynb\\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("C:\\Users\\Admin\\Documents\\machine_failure\\ipynb\\label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.stApp{

    background:#FFF8F0;

}

/* Hide Streamlit Menu */

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}



/* Sidebar */

section[data-testid="stSidebar"]{
    background:#FFF7EE;
}

/* Sidebar Text */

section[data-testid="stSidebar"] *{
    color:#5A4634;
}

/* Main Heading */

.main-title{

font-size:52px;

font-weight:700;

text-align:center;

color:#4E342E;

margin-bottom:5px;

}

/* Subtitle */

.sub-title{

font-size:22px;

text-align:center;

color:#D16B86;

margin-bottom:30px;

}

/* Hero Card */

.hero{

background:linear-gradient(
90deg,
#FFEFD5,
#FDEBD0,
#F8E8EE
);

padding:35px;

border-radius:25px;

box-shadow:0 10px 30px rgba(0,0,0,.08);

}

/* Cards */

.card{

background:#FFFFFF;

padding:25px;

border-radius:22px;

box-shadow:0 8px 20px rgba(0,0,0,.08);

border:1px solid #F6E8D5;

}

/* Result Success */

.success-card{

background:#ECFDF3;

padding:25px;

border-radius:18px;

border-left:8px solid #22C55E;

}

/* Result Failure */

.failure-card{

background:#FEF2F2;

padding:25px;

border-radius:18px;

border-left:8px solid #EF4444;

}

/* Gradient Button */

.stButton>button{

width:100%;

height:65px;

border:none;

border-radius:15px;

background:linear-gradient(
90deg,
#FFB5C5,
#FFD3A5
);

color:white;

font-size:22px;

font-weight:bold;

transition:.3s;

}

.stButton>button:hover{

transform:scale(1.02);

background:linear-gradient(
90deg,
#FFA9B9,
#FFC890
);

}

/* Metric Cards */

.metric{

background:white;

padding:20px;

border-radius:15px;

text-align:center;

box-shadow:0 5px 15px rgba(0,0,0,.05);

}

/* Footer */

.footer{

text-align:center;

font-size:17px;

color:#7A6653;

margin-top:30px;

}

</style>
""", unsafe_allow_html=True)
# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown(
        """
        <h1 style="text-align:center;color:#6B4F3B;">
        🏭
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<h2 style='text-align:center;color:#5A4634;'>Machine Dashboard</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📌 Project")

    st.success("Machine Failure Prediction")

    st.markdown("### 💻 Technologies")

    st.info("""
✅ Python

✅ Streamlit

✅ Scikit-Learn

✅ Random Forest

✅ Pickle
""")

    st.markdown("### 🤖 Model")

    st.write("Random Forest Classifier")

    st.markdown("---")

    st.markdown("### 📈 Accuracy")

    st.metric(
        label="Random Forest",
        value="98.40%"
    )

    st.markdown("---")


# =====================================================
# TITLE
# =====================================================

st.markdown(
"""
<div class="main-title">
⚙ Machine Failure Prediction System
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class="sub-title">
AI Powered Predictive Maintenance Dashboard
</div>
""",
unsafe_allow_html=True
)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown(
"""
<div class="hero">

<h2 style="color:#5A4634;">
🏭 Smart Machine Monitoring
</h2>

<p style="font-size:18px;color:#6F5C48;">

Predict industrial machine failures before they occur using
Artificial Intelligence and Machine Learning.

</p>

<br>

<table width="100%">
<tr>

<td align="center">
<h3>⚙</h3>
<b>Machine Health</b>
</td>

<td align="center">
<h3>📊</h3>
<b>Real-Time Analysis</b>
</td>

<td align="center">
<h3>🤖</h3>
<b>AI Prediction</b>
</td>

</tr>
</table>

</div>
""",
unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION TITLE
# =====================================================

st.markdown(
"""
<h1 style="
text-align:center;
color:#5A4634;
margin-bottom:20px;
">
📝 Enter Machine Details
</h1>
""",
unsafe_allow_html=True
)
# =====================================================
# INPUT SECTION
# =====================================================

col1, col2, col3 = st.columns(3)

# -----------------------
# COLUMN 1
# -----------------------
with col1:


    st.markdown("### 🏭 Machine Type")

    machine_type = st.selectbox("Machine Type",
        ["select", "L", "M", "H"],
        key="machine"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🌡 Air Temperature (K)")

    air_temp = st.number_input(
        "Air Temperature",
        min_value=250.0,
        max_value=400.0,
        value=250.0,
        step=0.1,
        key="air"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# COLUMN 2
# -----------------------
with col2:


    st.markdown("### 🔥 Process Temperature (K)")

    process_temp = st.number_input(
        "Process Tenperature",
        min_value=250.0,
        max_value=400.0,
        value=250.0,
        step=0.1,
        key="process"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### ⚙ Rotational Speed (RPM)")

    rotational_speed = st.number_input(
        "Rotational Speed",
        min_value=1000,
        max_value=3000,
        value=1000,
        step=1,
        key="rpm"
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------
# COLUMN 3
# -----------------------
with col3:


    st.markdown("### 🔩 Torque (Nm)")

    torque = st.number_input(
        "Torque",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.1,
        key="torque"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### ⏱ Tool Wear (min)")

    tool_wear = st.number_input(
        "Tool Wear",
        min_value=0,
        max_value=300,
        value=0,
        step=1,
        key="wear"
    )

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)

# =====================================================
# PREDICT BUTTON
# =====================================================


# =====================================================
# PREDICTION
# =====================================================

predict = st.button("🚀 Predict Machine Failure",    use_container_width=True
)

if predict:

    if machine_type == "select":
        st.warning("⚠ Please select a Machine Type before predicting.")
        st.stop()   # Stop execution here

    else:

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
        prediction = model.predict(input_scaled)

        if prediction[0] == 0:
            st.success("✅ Machine is Healthy")
        else:
            st.error("⚠ Machine Failure Predicted")

    # ============================
    # RESULT CARD
    # ============================

    if prediction[0] == 0:

        st.markdown(
        """
        <div class="success-card">

        <h2 style="color:#16A34A;text-align:center;">
        ✅ MACHINE IS HEALTHY
        </h2>

        <p style="text-align:center;
        font-size:18px;
        color:#444;">

        The machine is operating normally.
        No immediate maintenance is required.

        </p>

        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown(
        """
        <div class="failure-card">

        <h2 style="color:#DC2626;text-align:center;">
        ⚠ MACHINE FAILURE PREDICTED
        </h2>

        <p style="text-align:center;
        font-size:18px;
        color:#444;">

        The machine may fail soon.
        Preventive maintenance is recommended.

        </p>

        </div>
        """,
        unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================
    # INPUT SUMMARY
    # ============================

    st.markdown(
    """
    <h2 style="color:#5A4634;text-align:center;">
    📋 Machine Summary
    </h2>
    """,
    unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("🏭 Machine Type", machine_type)
        st.metric("🌡 Air Temperature", f"{air_temp} K")

    with c2:
        st.metric("🔥 Process Temperature", f"{process_temp} K")
        st.metric("⚙ Speed", f"{rotational_speed} RPM")

    with c3:
        st.metric("🔩 Torque", f"{torque} Nm")
        st.metric("⏱ Tool Wear", f"{tool_wear} min")

# =====================================================
# FOOTER
# =====================================================

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("---")

st.markdown(
"""
<div class="footer">

🏭 <b>Machine Failure Prediction System</b><br><br>

Developed using ❤️ Streamlit • Scikit-Learn • Random Forest

<br><br>

</div>
""",
unsafe_allow_html=True)
# =====================================================
# PREMIUM DASHBOARD
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<h2 style='text-align:center;
color:#5A4634;'>
📈 Machine Health Dashboard
</h2>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class='card'>
    <h1 style='text-align:center;'>🏭</h1>
    <h4 style='text-align:center;'>Machine</h4>
    <h2 style='text-align:center;color:#D97706;'>
    """ + machine_type + """
    </h2>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class='card'>
    <h1 style='text-align:center;'>🌡</h1>
    <h4 style='text-align:center;'>Air Temp</h4>
    <h2 style='text-align:center;color:#2563EB;'>
    {air_temp} K
    </h2>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class='card'>
    <h1 style='text-align:center;'>⚙</h1>
    <h4 style='text-align:center;'>RPM</h4>
    <h2 style='text-align:center;color:#16A34A;'>
    {rotational_speed}
    </h2>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class='card'>
    <h1 style='text-align:center;'>🔩</h1>
    <h4 style='text-align:center;'>Torque</h4>
    <h2 style='text-align:center;color:#DC2626;'>
    {torque}
    </h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# STATUS BAR
# =====================================================
if predict:
    st.subheader("Machine Status")

    if prediction[0] == 0:

       st.progress(100)

       st.success("Machine Health : Excellent")

    else:

        st.progress(35)

        st.error("Machine Health : Poor")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PARAMETERS
# =====================================================

st.subheader("Machine Parameters")

table = {
    "Parameter":[
        "Machine Type",
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear"
    ],
    "Value":[
        machine_type,
        air_temp,
        process_temp,
        rotational_speed,
        torque,
        tool_wear
    ]
}

st.table(table)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# ABOUT PROJECT
# =====================================================

with st.expander("About this Project"):

    st.write("""
This project predicts industrial machine failure
using Machine Learning.

Algorithm Used:

• Random Forest

Libraries Used:

• Streamlit

• Scikit-learn

• Pandas

• NumPy

• Pickle

Dataset:

AI4I 2020 Predictive Maintenance Dataset
""")