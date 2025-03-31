
# AI-Based Attack Detection System

## Overview
This project is an **AI-powered Attack Detection System** that utilizes **CatBoost**, a gradient boosting algorithm, to classify network attacks. It preprocesses network traffic data and trains a model to detect and classify attacks accurately.

## Machine Learning Model
- Uses **CatBoost Classifier** for training and classification
- Performs **hyperparameter tuning** via GridSearchCV

## Dynamic Test Data Handling
- Ensures compatibility between training and test datasets
- Handles missing and extra features dynamically

## Model Persistence
- Saves the trained model (`catboost_model.pkl`) for later use
- Stores the trained `StandardScaler` (`scaler.pkl`) for consistent feature scaling

## Prediction Handling
- Model is capable of making predictions even if 80% of the expected features are available, but accuracy may vary.

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
1. **Prepare your dataset**: Ensure your CSV file (`final_dataset.csv`) is formatted correctly with a `label` column for classification.
2. **Run the model training script**:
   ```sh
   python train_model.py
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

## License
This project is licensed under the MIT License.
```

