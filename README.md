# Machine Failure Prediction System

## Overview

The Machine Failure Prediction System is a Machine Learning web application that predicts whether an industrial machine is likely to fail based on its operating conditions. The project helps industries perform predictive maintenance by identifying potential machine failures before they occur, reducing downtime and maintenance costs.

The application is developed using Python, Scikit-learn, and Streamlit with a Random Forest Classifier trained on the AI4I 2020 Predictive Maintenance dataset.

---

## Features

- Predicts machine failure using operational parameters
- User-friendly Streamlit web interface
- Data preprocessing and feature engineering
- Random Forest Machine Learning model
- Model serialization using Pickle
- Modular project structure
- Fast and accurate predictions

---

## Dataset

Dataset Used:
AI4I 2020 Predictive Maintenance Dataset

Input Features:
- Machine Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

Target:
- Machine Failure
  - 0 → No Failure
  - 1 → Failure

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Pickle

---

## Project Structure

```
Machine-Failure-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── data/
│   └── machine_failure.csv
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predictor.py
│   ├── save_model.py
│   └── visual.py
│
└── images/
    └── screenshot.png
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Machine-Failure-Prediction.git
```

Go to the project directory

```bash
cd Machine-Failure-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Model Used

Random Forest Classifier

The model was trained after:

- Data Cleaning
- Label Encoding
- Feature Scaling
- Train-Test Split
- Model Training
- Model Evaluation

---

## Future Enhancements

- Deep Learning Models
- Real-time IoT Sensor Integration
- Cloud Deployment
- Maintenance Scheduling
- Failure Explanation Dashboard

---

## Author

Barisha B

M.Sc. Computer Science

Machine Learning Project
