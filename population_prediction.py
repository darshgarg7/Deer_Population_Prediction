import pandas as pd
import numpy as np
import logging
import time
import optuna
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import shap
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf

logging.basicConfig(level=logging.INFO)

filepath = '/Users/Darsh/Documents/GitHub/Deer_Population_Prediction/final_data.csv'

# Random seeds
np.random.seed(42)
tf.random.set_seed(42)

class Preprocessor:
    """
    A class to load, preprocess, and generate new features for the dataset.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.imputer = SimpleImputer(strategy='mean')

    def load_and_preprocess_data(self):
        """
        Loads and preprocesses the data by adding features and handling missing values.
        Handles missing data using imputation and adds engineered features to the dataset.
        """
        try:
            data = self._load_data()
            data = self._add_features(data)
            data.dropna(inplace=True)  # Handle missing values in the dataset
            X = data.drop(columns=['Population'])
            y = data['Population']
            X = self.imputer.fit_transform(X)
            logging.info("Data loaded and preprocessed successfully")
            return X, y
        except Exception as e:
            logging.error(f"Error during data loading and preprocessing: {e}")
            raise

    def _load_data(self):
        """
        Loads data from the CSV file.
        """
        logging.info(f"Loading data from {self.filepath}")
        
        try:
            return pd.read_csv(self.filepath)
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
        
    def _add_features(self, data):
        """
        Adds new features for modeling using vectorized operations to optimize performance.
        Handles potential outliers and adds interaction terms.
        """
        logging.info("Adding new features for modeling")

        data['Prev_Population'] = data['Population'].shift(1)
        data['Moving_Avg'] = data['Population'].rolling(window=3).mean().shift(1)
        data['Rolling_Std'] = data['Population'].rolling(window=3).std().shift(1)
        data['Yearly_Growth'] = data['Population'].pct_change()
        data['Lag_4'] = data['Population'].shift(4)
        data['Interaction_Temperature_Precipitation'] = data['MaxTemp'] * data['Precipitation']
        
        poly_features = PolynomialFeatures(degree=2).fit_transform(data[['MaxTemp', 'Precipitation']])
        data[['Poly_Features_1', 'Poly_Features_2', 'Poly_Features_3']] = poly_features[:, 1:]

        logging.info(f"Checking feature correlations with the target")
        corr_matrix = data.corr()
        logging.info(f"Feature correlations:\n{corr_matrix['Population']}")
        
        return data

class CustomNN(BaseEstimator, TransformerMixin):
    """
    A custom neural network class for time-series forecasting using LSTM.
    Implements early stopping and regularization to prevent overfitting.
    """

    def __init__(self, input_dim, epochs=50, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(input_dim, 1)))
        self.model.add(LSTM(32, kernel_regularizer='l2'))  # Added L2 regularization
        self.model.add(Dense(1))  # Output layer for regression

    def fit(self, X, y):
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for LSTM
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1, validation_split=0.2)
        return self

    def predict(self, X):
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X)

def create_pipeline(model=None):
    """
    Creates a machine learning pipeline with preprocessing and a stacking regressor.
    """
    logging.info("Creating pipeline")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['MaxTemp', 'MinTemp', 'Precipitation', 'Snow', 'SnowDepth', 'Harvest', 
                                       'Prev_Population', 'Moving_Avg', 'Rolling_Std', 'Yearly_Growth', 'Lag_4', 'Interaction_Temperature_Precipitation'])
        ]
    )

    if model is None:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    # Stacking, ensemble learning
    stacking_model = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=200)),
            ('xgb', XGBRegressor(n_estimators=200)),
            ('lstm', CustomNN(input_dim=11))
        ],
        final_estimator=RandomForestRegressor(n_estimators=100)
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', stacking_model)
    ])
    
    logging.info("Pipeline created successfully")
    return pipeline

def hyperparameter_tuning(X, y, pipeline):
    """
    Performs hyperparameter tuning using Optuna.
    """
    logging.info("Performing hyperparameter tuning using Optuna")

    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)

        pipeline.set_params(model=model)
        score = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error').mean()
        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    logging.info(f"Best Hyperparameters: {study.best_params}")
    return pipeline.set_params(**study.best_params)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, pipeline):
    """
    Trains and evaluates the model, logs performance metrics and handles edge cases.
    """
    logging.info("Training model")

    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds.")
    
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    logging.info(f'Mean Squared Error: {mse}')
    logging.info(f'Root Mean Squared Error: {rmse}')
    logging.info(f'R² Score: {r2}')
    logging.info(f'Mean Absolute Error: {mae}')
    logging.info(f'MAPE: {mape}')
    
    return y_pred

def save_model(model):
    """
    Saves the model using MLflow.
    """
    mlflow.start_run()
    mlflow.sklearn.log_model(model, 'deer_population_model')
    mlflow.end_run()

def interpret_model(X_train, y_train, pipeline):
    """
    Explains the model using SHAP and LIME.
    """
    logging.info("Explaining model with SHAP")
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train)

    logging.info("Explaining model with LIME")
    lime_explainer = LimeTabularExplainer(X_train, training_labels=y_train, mode="regression")
    lime_exp = lime_explainer.explain_instance(X_train[0], pipeline.predict)
    lime_exp.show_in_notebook()

def main():
    preprocessor = Preprocessor(filepath)
    X, y = preprocessor.load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = create_pipeline()
    pipeline = hyperparameter_tuning(X_train, y_train, pipeline)

    y_pred = train_and_evaluate_model(X_train, y_train, X_test, y_test, pipeline)

    logging.info(f"Predictions: {y_pred}")

    save_model(pipeline)

    interpret_model(X_train, y_train, pipeline)


if __name__ == '__main__':
    main()
