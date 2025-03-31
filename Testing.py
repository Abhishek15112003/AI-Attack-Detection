import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# Load the saved model and scaler
model_path = r"C:\Users\abhis\OneDrive\Desktop\DAA\catboost_model.pkl"
scaler_path = r"C:\Users\abhis\OneDrive\Desktop\DAA\scaler.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

with open(scaler_path, "rb") as file:
    scaler = pickle.load(file)

# Load new test dataset
test_data_path = r"C:\Users\abhis\OneDrive\Desktop\DAA\test_dataset.csv"
test_df = pd.read_csv(test_data_path)

# Define the selected features used in training
selected_features = model.feature_names_

# Function to preprocess test dataset
def preprocess_test_data(test_df, expected_features, scaler):
    test_df = test_df.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = test_df.select_dtypes(include=["object"]).columns
    numerical_cols = test_df.select_dtypes(include=["float64", "int64"]).columns
    
    # Encode categorical features
    test_df = pd.get_dummies(test_df, columns=categorical_cols)
    
    # Ensure test data has the same columns as training
    missing_cols = set(expected_features) - set(test_df.columns)
    for col in missing_cols:
        test_df[col] = 0  # Add missing columns with default value 0
    
    # Keep only relevant features and reorder columns
    test_df = test_df[expected_features]
    
    # Scale numerical features
    num_cols = list(set(scaler.feature_names_in_) & set(test_df.columns))
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    return test_df

# Preprocess test data
processed_test_df = preprocess_test_data(test_df, selected_features, scaler)

# Make predictions
y_pred = model.predict(processed_test_df)

# Save predictions
output_path = r"C:\Users\abhis\OneDrive\Desktop\DAA\predictions.csv"
pd.DataFrame(y_pred, columns=["Predicted Label"]).to_csv(output_path, index=False)

print(f"Predictions saved at {output_path}")