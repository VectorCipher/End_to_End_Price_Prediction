import logging
from abc import ABC , abstractmethod
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error,r2_score
import pandas as pd

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s ")
## Abstract class for the Model evaluation to implement evaluate model method
class ModelEvaluationStrategy(ABC):
    @abstractmethod
    def evaluate_model(self,model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series)-> dict:
        pass
    
## Here we have created a class to calculate mse and R2 score 
class RegressionModelEvaluationStrategy(ModelEvaluationStrategy):
    def evaluate_model(self, model:RegressorMixin, X_test:pd.DataFrame, y_test:pd.Series)-> dict:
        logging.info("Predicting using the trained model.")
        y_pred = model.predict(X_test)
        
        logging.info("Calculating Evaluation Metrics.")
        mse = mean_squared_error(y_test,y_pred)
        r2=r2_score(y_test,y_pred)
        
        metrics = {"Mean Squared Error":mse,
                   "R2_Score": r2,
                   }
        logging.info(f"Model Evaluation Metrics:{metrics}")
        return metrics
    
## Here is the context class for model evaluation    
class ModelEvaluator:
    def __init__(self,strategy:ModelEvaluationStrategy):
        self.strategy = strategy
        
    def set_strategy(self,strategy:ModelEvaluationStrategy):
        logging.info("Switching model Evaluation Strategy.")
        self.strategy = strategy
        
    def evaluate(self, model:RegressorMixin,X_test:pd.DataFrame,y_test:pd.Series):
        logging.info("Evaluating the model")
        self.strategy.evaluate_model(model,X_test,y_test)