# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import pickle


# =====================================================
# SAVE MODEL FUNCTION
# =====================================================

def save_models(model, scaler, encoder):

    # Create Folder
    os.makedirs("artifacts/model", exist_ok=True)

    # Save Random Forest Model
    with open(
        "artifacts/model/machine_failure_prediction_model.pkl",
        "wb"
    ) as file:

        pickle.dump(model, file)

    # Save Scaler
    with open(
        "artifacts/model/scaler.pkl",
        "wb"
    ) as file:

        pickle.dump(scaler, file)

    # Save Label Encoder
    with open(
        "artifacts/model/label_encoder.pkl",
        "wb"
    ) as file:

        pickle.dump(encoder, file)

    print("=" * 50)
    print("Models Saved Successfully!")
    print("Location : artifacts/model/")
    print("=" * 50)