from abc import ABC , abstractmethod
import pandas as pd 


## Data Inspection Strategy 
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self,df:pd.DataFrame):
        pass


## Inspect the data types of each column and count the Non null values
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self,df:pd.DataFrame):
        print(df.info())## It will provide a concise summary of DataFrame 


## df.describe() provides descriptive statistics such as count, mean, standard deviation, quartiles, and range for numerical columns in a DataFrame.
class SummaryStatisticsInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df:pd.DataFrame):
        print("\n Summary Statistics(Numerical Features):")
        print(df.describe())
        print("\n Summary Statistics(Categorical features):")
        print(df.describe(include="object"))

## We want which DataInspector to use     
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self.strategy = strategy
        
    def set_strategy(self,strategy:DataInspectionStrategy):
        self.strategy=strategy
        
    def execute_strategy(self,df:pd.DataFrame):
        self.strategy.inspect(df)
        
if __name__ == "__main__":
    ## Load the data
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    ## Initialize the Data Inspector with a specific strategy
    inspector = DataInspector(DataTypesInspectionStrategy())
    inspector.execute_strategy(df)
    ## Change the strategy
    inspector.set_strategy(SummaryStatisticsInspectionStrategy())
    inspector.execute_strategy(df)