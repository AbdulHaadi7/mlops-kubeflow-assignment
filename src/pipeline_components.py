import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from kfp.dsl import component
import subprocess

# 1. Data Extraction
@component(base_image="python:3.11")
def extract_data(dvc_path: str, output_path: str):
    """
    Fetch dataset from DVC remote storage.
    """
    subprocess.run(f"dvc get {dvc_path} data/raw_data.csv -o {output_path}", shell=True, check=True)
    print(f"Dataset saved to {output_path}")

# 2. Data Preprocessing
@component(base_image="python:3.11")
def preprocess_data(input_csv: str, X_train_csv: str, X_test_csv: str, y_train_csv: str, y_test_csv: str):
    """
    Load CSV, scale features, split into train/test, and save.
    """
    df = pd.read_csv(input_csv)
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    pd.DataFrame(X_train, columns=X.columns).to_csv(X_train_csv, index=False)
    pd.DataFrame(X_test, columns=X.columns).to_csv(X_test_csv, index=False)
    pd.DataFrame(y_train, columns=['PRICE']).to_csv(y_train_csv, index=False)
    pd.DataFrame(y_test, columns=['PRICE']).to_csv(y_test_csv, index=False)

# 3. Model Training
@component(base_image="python:3.11")
def train_model(X_train_csv: str, y_train_csv: str, model_path: str):
    """
    Train Random Forest model and save it.
    """
    X_train = pd.read_csv(X_train_csv)
    y_train = pd.read_csv(y_train_csv)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

# 4. Model Evaluation
@component(base_image="python:3.11")
def evaluate_model(model_path: str, X_test_csv: str, y_test_csv: str, metrics_path: str):
    """
    Evaluate model and save metrics to CSV.
    """
    X_test = pd.read_csv(X_test_csv)
    y_test = pd.read_csv(y_test_csv)
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    metrics = {'MSE': mse, 'R2': r2}
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
