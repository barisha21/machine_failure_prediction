from src.data_loader import load_dataset
from src.data_preprocessing import clean_data
from src.feature_engineering import feature_engineering
from src.visualization import save_all_plots
from src.train_model import train_models
from src.save_model import save_models


def main():

    print("="*60)
    print(" MACHINE FAILURE PREDICTION PROJECT ")
    print("="*60)

    # Dataset Path
    dataset_path = r"data/machine failure.csv"

    # Load Dataset
    df = load_dataset(dataset_path)

    print("Dataset Loaded Successfully")

    # Data Preprocessing
    df = clean_data(df)

    print("Data Cleaning Completed")

    # Save EDA Plots
    save_all_plots(df)

    print("EDA Plots Saved")

    # Feature Engineering
    X_scaled, y, scaler, encoder = feature_engineering(df)

    print("Feature Engineering Completed")

    # Train Model
    model = train_models(X_scaled, y)

    print("Model Training Completed")

    # Save Model
    save_models(model, scaler, encoder)

    print("Model Saved Successfully")

    print("="*60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    main()