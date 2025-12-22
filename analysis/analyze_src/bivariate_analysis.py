from abc import ABC, abstractmethod
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Abstract class to implement analyze method in every class 
class BiVariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df:pd.DataFrame,feature1:str,feature2:str):
        pass

## This is the class for num vs num analysis 
## between two numerical features
class NumericalvsNumericalAnalysis(BiVariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature1:str, feature2:str):
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} V/S {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        
class CategoricalvsNumericalAnalysis(BiVariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature1:str, feature2:str):
        plt.figure(figsize=(10,6))
        sns.boxplot(x=feature1,y=feature2,data=df)
        plt.title(f"{feature1} V/s {feature2} ")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show() 
        
class BiVariateAnalyzer:
    def __init__(self,strategy:BiVariateAnalysisStrategy):
        self.strategy = strategy
        
    def set_strategy(self,strategy:BiVariateAnalysisStrategy):
        self.strategy=strategy
        
    def execute_strategy(self,df:pd.DataFrame,feature1:str,feature2:str):
        self.strategy.analyze(df,feature1,feature2)
        

if __name__ == "__main__":
    ## Load the dataset
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    ## Analyzing relationship between numerical features
    analyzer = BiVariateAnalyzer(NumericalvsNumericalAnalysis())
    analyzer.execute_strategy(df,"Gr Liv Area","SalePrice")
    ##Analyzing relationship between a categorical and numerical Feature
    analyzer.set_strategy(CategoricalvsNumericalAnalysis())
    analyzer.execute_strategy(df,"Overall Qual","SalePrice")
