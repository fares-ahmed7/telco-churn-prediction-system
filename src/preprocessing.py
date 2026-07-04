import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data(train_path=TRAIN_PATH):
    """
    Load dataset and separate features from target.
    """

    df = pd.read_csv(train_path)

    # Drop unnecessary column
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Replace special values
    replace_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in replace_cols:
        df[col] = df[col].replace({
            "No internet service": "No",
            "No phone service": "No"
        })

    # Binary Encoding

    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def handle_missing_values(X_train, X_test, strategy="median"):
    """
    Impute missing numerical values using statistics
    learned from the training data.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_cols = X_train.select_dtypes(include="number").columns

    imputer = SimpleImputer(strategy=strategy)

    X_train[numeric_cols] = imputer.fit_transform(
        X_train[numeric_cols]
    )

    X_test[numeric_cols] = imputer.transform(
        X_test[numeric_cols]
    )

    return X_train, X_test, imputer


def encode_categorical_features(X_train, X_test):
    """
    Apply One-Hot Encoding to categorical features.
    """

    categorical_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ]

    X_train = pd.get_dummies(
        X_train,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )

    X_train, X_test = X_train.align(
        X_test,
        join="left",
        axis=1,
        fill_value=0
    )

    return X_train, X_test


def scale_features(X_train, X_test):
    """
    Standardize numerical features.
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, scaler