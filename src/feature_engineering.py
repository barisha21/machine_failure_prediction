from sklearn.preprocessing import LabelEncoder, StandardScaler


def feature_engineering(df):

    # Label Encoding
    encoder = LabelEncoder()

    df["Type"] = encoder.fit_transform(df["Type"])

    # Feature Selection
    X = df.drop("Machine failure", axis=1)
    y = df["Machine failure"]

    # Feature Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoder