import logging
from abc import ABC ,abstractmethod
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

logging.basicConfig(level = logging.INFO,format = "%(asctime)s - %(levelname)s - %(message)s ",force=True)

class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_and_train_model(self,X_train:pd.DataFrame,y_train:pd.Series)-> RegressorMixin:
        pass
    
class LinearRegressionStrategy(ModelBuildingStrategy):
    
    def build_and_train_model(self, X_train:pd.DataFrame, y_train:pd.Series) -> Pipeline:
        if not isinstance(X_train,pd.DataFrame):
            raise TypeError("X_train must be a pandas DataFrame.")
        if not isinstance(y_train,pd.Series):
            raise TypeError("y_train must be a pandas series.")
        
        logging.info("Initializing Linear Regression Model with scaling")
        
        pipeline = Pipeline(
            [
            ("scalar", StandardScaler()),
            ("model", LinearRegression()),
            ]
        )
        logging.info("Training Linear Regression Model.")
        pipeline.fit(X_train,y_train)
        
        logging.info("Model training Completed.")
        return pipeline
    
class ModelBuilder:
    def __init__(self,strategy:ModelBuildingStrategy):
        self.strategy = strategy
        
    def set_strategy(self,strategy:ModelBuildingStrategy):
        logging.info("Changing model building strategy")
        self.strategy=strategy
    def build_model(self,X_train:pd.DataFrame,y_train:pd.Series)-> RegressorMixin:
        logging.info("Building and trainig the model using the selected strategy.")
        return self.strategy.build_and_train_model(X_train,y_train)
    
    
if __name__ == "__main__":
    df = pd.read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/extracted_data/AmesHousing.csv")
    X_train = df.drop(columns=["SalePrice"])
    y_train = df["SalePrice"]
    
    model_builder = ModelBuilder(LinearRegressionStrategy())
    trained_model = model_builder.build_model(X_train,y_train)
    print(trained_model.named_steps["model"].coef_)