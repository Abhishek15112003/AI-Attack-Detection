import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(r"C:\Users\abhis\OneDrive\Desktop\DAA\final_dataset.csv")  # Update file path

# Define features and target
X = df.drop(columns=["label"])  # Features
y = df["label"]  # Target

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns

# Encode categorical features
X = pd.get_dummies(X, columns=categorical_cols)

# Standardize numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save the scaler for later use
scaler_save_path = r"C:\Users\abhis\OneDrive\Desktop\DAA\scaler.pkl"
with open(scaler_save_path, "wb") as file:
    pickle.dump(scaler, file)
print(f"Scaler saved at {scaler_save_path}")

# Train an initial CatBoost model for feature importance
catboost = CatBoostClassifier(verbose=0)
catboost.fit(X, y)

# Get feature importance
feature_importance = catboost.get_feature_importance()
feature_names = X.columns

# Convert to DataFrame for filtering
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

# Normalize importance values (convert to percentage)
feature_importance_df['Importance'] = feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()

# Keep only features with importance >= 1%
selected_features = feature_importance_df[feature_importance_df['Importance'] >= 0.01]['Feature'].tolist()

# Remove unimportant features
X = X[selected_features]

# Print remaining feature count
print(f"Remaining features after filtering: {len(selected_features)}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CatBoost classifier for final model training
catboost = CatBoostClassifier(verbose=0)

# Hyperparameter tuning
param_grid = {
    "iterations": [100, 500],
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1]
}

grid_search = GridSearchCV(catboost, param_grid, scoring="accuracy", cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Save the trained model as a .pickle file
model_save_path = r"C:\Users\abhis\OneDrive\Desktop\DAA\catboost_model.pkl"
with open(model_save_path, "wb") as file:
    pickle.dump(best_model, file)
print(f"Model saved at {model_save_path}")

# Function to preprocess test dataset dynamically
def preprocess_test_data(test_df, expected_features, scaler):
    """
    Preprocesses the test dataset:
    - Drops extra columns
    - Handles missing columns if at least 75% of expected columns are present
    """
    test_df = test_df.copy()

    # Retain only the features used during training
    test_df = test_df.loc[:, test_df.columns.intersection(expected_features)]

    # Find missing columns
    missing_columns = list(set(expected_features) - set(test_df.columns))

    # If too many features are missing, reject the test data
    if len(missing_columns) > 0.25 * len(expected_features):
        raise ValueError("Insufficient valid columns in test data.")

    # Add missing columns as NaN
    for col in missing_columns:
        test_df[col] = np.nan

    # Ensure column order matches training
    test_df = test_df[expected_features]

    # Apply standard scaling to numerical columns
    num_cols = list(set(scaler.feature_names_in_) & set(test_df.columns))
    test_df[num_cols] = scaler.transform(test_df[num_cols])

    return test_df

# Process test dataset dynamically before making predictions
try:
    processed_test_df = preprocess_test_data(X_test, selected_features, scaler)
    y_pred = best_model.predict(processed_test_df)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Model Accuracy: {accuracy:.4f}")

except ValueError as e:
    print(f"Error: {e}")