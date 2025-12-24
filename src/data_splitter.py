import logging 
from abc import ABC , abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level = logging.INFO,format="%(asctime)s - %(levelname)s - %(message)s",force=True)

##Abstract class to implement train test split 
class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self,df:pd.DataFrame,target_column:str):
        pass
    
## Class to implement simple train test split using sklearn

class SimpleTrainTestSplit(DataSplittingStrategy):
    def __init__(self,test_size = 0.25,random_state = 42):
        self.test_size=test_size
        self.random_state = random_state
        
    def split_data(self,df:pd.DataFrame,target_column:str):
        logging.info("Performing simple train-test split.")
        X=df.drop(columns=[target_column])
        y=df[target_column] 
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state) 
        logging.info("Train-test split completed.")
        return X_train,X_test,y_train,y_test
    
    
class DataDSplitter:
    def __init__(self,strategy:DataSplittingStrategy):
        self.strategy = strategy
        
    def set_strategy(self,strategy:DataSplittingStrategy):
        logging.info("Switching data splitting strategy.")
        self.strategy =strategy
        
    def split(self,df:pd.DataFrame,target_column:str):
        logging.info("Splitting the Data.")
        return self.strategy.split_data(df,target_column)

if __name__ == "__main__":
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    data_splitter = DataDSplitter(SimpleTrainTestSplit(test_size=0.25,random_state=42))
    X_train,X_test,y_train,y_test = data_splitter.split(df,target_column="SalePrice")