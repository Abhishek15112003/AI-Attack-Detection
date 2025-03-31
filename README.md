# AI-Based Attack Detection

## Overview
This project is an AI-powered model that detects the type of cyber attack occurring in a network. The model uses machine learning techniques to analyze network traffic and classify attacks. The dataset is preprocessed, important features are selected, and a CatBoostClassifier is trained to achieve high accuracy in attack detection.

## Features
- Loads and preprocesses a network attack dataset
- Encodes categorical features and scales numerical features
- Selects the most important features using feature importance analysis
- Trains a **CatBoostClassifier** for attack classification
- Hyperparameter tuning with GridSearchCV for optimal performance
- Saves the trained model and scaler for future use
- Supports dynamic test data preprocessing to handle missing columns

## Installation
To run this project, install the required dependencies using the following command:

```sh
pip install -r requirements.txt
```

## Usage
### 1. Train the Model
Ensure the dataset is available at the correct path, then run the training script:

```sh
python Network.py
```

This will preprocess the data, train the model, and save it as `catboost_model.pkl`.

### 2. Preprocess New Test Data
To process new test data and make predictions:

```python
from model import preprocess_test_data, best_model
import pandas as pd
import pickle

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load new test data
new_data = pd.read_csv("path_to_test_data.csv")

# Preprocess test data
processed_data = preprocess_test_data(new_data, selected_features, scaler)

# Make predictions
predictions = best_model.predict(processed_data)
print(predictions)
```

## Files
- **model.py** - Main script for training and saving the model
- **scaler.pkl** - Saved scaler for preprocessing test data
- **catboost_model.pkl** - Trained machine learning model
- **requirements.txt** - List of dependencies

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- CatBoost

## License
This project is open-source .
