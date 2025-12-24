from abc import ABC , abstractmethod 
import pandas as pd 
import logging

## logging is critical for Debugging Monitoring data changes Understanding pipeline behaviour
logging.basicConfig(level=logging.INFO,format="%(asctime)s- %(levelname)s - %(message)s",force=True)


## This class will act as a abstract calss to implement method handle
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self,df:pd.DataFrame) -> pd.DataFrame:
        pass


## Axis = 0 means we have to drop whole axis from the dataframe
## threshold_value = means for not-NA values Rows/Columns should be dropped
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self,axis=0,thresh=None):
        self.axis=axis
        self.thresh = thresh
    
    def handle(self,df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Dropping missing values with axis={self.axis} and threshold = {self.thresh}")
        df_cleaned = df.dropna(axis = self.axis, thresh=self.thresh)
        logging.info("Missing Values dropped.")
        return df_cleaned
    

## This is a class for which method u should handle missing values
class FillMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self,method="mean",fill_value = None):
        self.method = method
        self.fill_value = fill_value
        
    def handle(self,df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"filling missing values using method :{self.method}")
        df_cleaned = df.copy()
        
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns]=df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0],inplace=True)
        elif self.method ==  "constant":
            df_cleaned=df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(f"Unknown method {self.method}. No missing values is handled ")
        
        logging.info("Missing values Filled.")
        return df_cleaned

class MissingValueHandler:
    def __init__(self,strategy:MissingValueHandlingStrategy):
        self.startegy = strategy
        
    def set_strategy(self,strategy:MissingValueHandlingStrategy):
        self.strategy = strategy
        logging.info(f"Set Strategy to the {self.strategy}")
        
    def handle_missing_values(self,df:pd.DataFrame) -> pd.DataFrame:
        logging.info("Executing missing values handling startegy.")
        return self.startegy.handle(df)
    
if __name__ == "__main__":
    ## load the dataset 
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    ## Initialize missing value handler with a strategy
    missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis = 0,thresh = 3))
    df_cleaned = missing_value_handler.handle_missing_values(df)
    ##Switch the strategy
    missing_value_handler.set_strategy(FillMissingValuesStrategy(method="mean"))
    df_filled = missing_value_handler.handle_missing_values(df)
    
    