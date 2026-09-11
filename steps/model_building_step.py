import pandas as pd
from sklearn.pipeline import Pipeline
from src.model_building import LinearRegressionStrategy, ModelBuilder
from zenml import step


@step
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    builder = ModelBuilder(LinearRegressionStrategy())
    return builder.build_model(X_train, y_train)
