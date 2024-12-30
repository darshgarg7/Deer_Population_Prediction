import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from population_prediction import FeatureEngineer, create_pipeline, hyperparameter_tuning
from sklearn.exceptions import NotFittedError

@pytest.fixture
def load_data():
    """Fixture to load a mock dataset for testing purposes."""
    data = {
        'Population': [100, 200, 300, 400, 500],
        'MaxTemp': [15, 20, 25, 30, 35],
        'MinTemp': [5, 10, 15, 20, 25],
        'Precipitation': [0, 0.1, 0.2, 0.3, 0.4],
        'Snow': [0, 1, 2, 3, 4],
        'SnowDepth': [0, 0.5, 1, 1.5, 2],
        'Harvest': [10, 12, 14, 16, 18]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns='Population')
    y = df['Population']
    return X, y


def test_feature_engineer(load_data):
    """Test the FeatureEngineer class for adding new features to the dataset."""
    X = load_data
    feature_engineer = FeatureEngineer()
    X_transformed = feature_engineer.fit_transform(X)

    # Check if new features have been added
    assert 'Prev_Population' in X_transformed.columns, \
        "Prev_Population feature should be added"
    assert 'Moving_Avg' in X_transformed.columns, \
        "Moving_Avg feature should be added"

    # Ensure the original features are still present
    assert 'MaxTemp' in X_transformed.columns, \
        "MaxTemp should still be present in the transformed dataset"


def test_create_pipeline():
    """Test the creation of the machine learning pipeline."""
    pipeline = create_pipeline()

    # Ensure pipeline is a valid sklearn pipeline
    assert isinstance(pipeline, Pipeline), \
        "The pipeline should be an instance of sklearn's Pipeline"

    # Check the presence of expected steps in the pipeline
    assert 'preprocessor' in pipeline.named_steps, \
        "'preprocessor' step should be present in the pipeline"
    assert 'model' in pipeline.named_steps, \
        "'model' step should be present in the pipeline"


def test_train_and_evaluate_model(load_data):
    """Test the training and evaluation of the model."""
    X, y = load_data
    pipeline = create_pipeline()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    assert mse < 100, f"Model performance is poor, MSE is {mse}"

    assert y_pred.shape == y_test.shape, \
        "The shape of predicted values should match the shape of the target"


def test_hyperparameter_tuning(load_data):
    """Test hyperparameter tuning functionality."""
    X, y = load_data
    pipeline = create_pipeline()

    best_model = hyperparameter_tuning(X, y, pipeline)

    assert best_model is not None, \
        "Hyperparameter tuning should return a best model"
    assert hasattr(best_model, 'predict'), \
        "The best model should have a predict method"

    best_mse = mean_squared_error(y, best_model.predict(X))
    assert best_mse < 100, \
        f"Hyperparameter tuning did not improve model performance, MSE is {best_mse}"


def test_feature_engineer_missing_data():
    """Test if the FeatureEngineer class handles missing data correctly."""
    data = {
        'Population': [100, 200, 300, None, 500],
        'MaxTemp': [15, 20, 25, 30, None],
        'MinTemp': [5, 10, 15, 20, 25],
        'Precipitation': [0, 0.1, 0.2, 0.3, 0.4],
        'Snow': [0, 1, 2, 3, 4],
        'SnowDepth': [0, 0.5, 1, None, 2],
        'Harvest': [10, 12, 14, 16, 18]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns='Population')
    feature_engineer = FeatureEngineer()

    X_transformed = feature_engineer.fit_transform(X)

    assert X_transformed.isnull().sum().sum() == 0, \
        "There should be no missing values after feature engineering"


def test_empty_dataset():
    """Test if the model handles an empty dataset gracefully."""
    X_empty = pd.DataFrame()
    y_empty = pd.Series()

    pipeline = create_pipeline()

    with pytest.raises(ValueError):
        pipeline.fit(X_empty, y_empty)


def test_predict_without_fit(load_data):
    """Test if an error is raised when predicting without fitting the model."""
    X, y = load_data
    pipeline = create_pipeline()

    with pytest.raises(NotFittedError):
        pipeline.predict(X)


if __name__ == '__main__':
    pytest.main()
