from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
import pandas as pd


def train_models(X, y):

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Initialize Models
    lr_model = LogisticRegression(random_state=42)

    dt_model = DecisionTreeClassifier(random_state=42)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    svm_model = SVC(
        kernel="rbf",
        random_state=42
    )

    # Train Models
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # Predictions
    lr_pred = lr_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)

    # Accuracy
    results = pd.DataFrame({

        "Model":[
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "SVM"
        ],

        "Accuracy":[
            accuracy_score(y_test, lr_pred),
            accuracy_score(y_test, dt_pred),
            accuracy_score(y_test, rf_pred),
            accuracy_score(y_test, svm_pred)
        ]

    })

    print(results)

    return rf_model