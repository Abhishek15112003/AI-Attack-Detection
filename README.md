
# AI-Based Attack Detection System

## Overview
This project is an **AI-powered Attack Detection System** that utilizes **CatBoost**, a gradient boosting algorithm, to classify network attacks. It preprocesses network traffic data and trains a model to detect and classify attacks accurately.

## Machine Learning Model
- Uses **CatBoost Classifier** for training and classification
- Performs **hyperparameter tuning** via GridSearchCV
- Trained model is saved as `catboost_model.pkl`
- Preprocessing is handled using `scaler.pkl` for consistent feature scaling

## Dataset
- The dataset (`final_dataset.csv`) is already preprocessed.
- The model is capable of making predictions even if 80% of the expected features are available, though accuracy may vary.

## Installation
### Prerequisites
Ensure you have **Python 3.7+** installed along with the required dependencies.

### Setup
1. **Clone the repository**:
   ```sh
   git clone https://github.com/Abhishek15112003/AI-Attack-Detection.git
   cd AI-Attack-Detection
   ```



2. **Install required packages**:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Ensure required files are available**:
   - `catboost_model.pkl` (trained model)
   - `scaler.pkl` (preprocessing scaler)
   - `final_dataset.csv` (preprocessed dataset)

2. **Run the model training script**:
   ```sh
   python Network.py
   ```
   This will preprocess data, train the model, and save it as `catboost_model.pkl`.

3. **Use the trained model for predictions**: Load the model and process new data dynamically.

## Requirements
Ensure the following Python packages are installed:
```sh
numpy
pandas
scikit-learn
catboost
```  
You can install them with:
```sh
pip install -r requirements.txt
```

## Future Enhancements
- Improve model accuracy using additional feature selection techniques.
- Deploy the model as an API for real-time attack detection.
- Integrate visualization tools for monitoring network traffic.

## Author
[Abhishek Anjana](https://github.com/Abhishek15112003)

