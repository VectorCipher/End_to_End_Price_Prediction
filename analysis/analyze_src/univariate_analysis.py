from abc import ABC ,abstractmethod
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


## This is for the one variable analysis in the dataframe 
## df -> DataFrame and Feature -> column or variable to analyze 
## returns the visual distribution of the feature.
class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self,df:pd.DataFrame,feature:str):
        pass
    
## Histogram gives us the freq distribution of a numerical feature by grouping values into intervals called bins.
## KDE refers to Kernel Density Estimation
## It refers to smooth continuous estimate of the Probabbility density function of a numerical feature.
class NumericalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame , feature:str):
        plt.figure(figsize=(10,6))
        sns.histplot(df[feature],kde=True,bins=30)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()

## We do categorical analysis as for a particular feature which category occur more nu. of times
class CategoricalUnivariateAnalysis(UnivariateAnalysisStrategy):
    def analyze(self, df:pd.DataFrame, feature:str):
        plt.figure(figsize=(10,6))
        sns.countplot(x=feature,data=df,palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()
        
class UnivariateAnalyzer:
    def __init__(self,strategy:UnivariateAnalysisStrategy):
        self.strategy = strategy
        
    def set_strategy(self,startegy:UnivariateAnalysisStrategy):
        self.strategy = startegy
        
    def execute_strategy(self,df:pd.DataFrame,feature:str):
        self.strategy.analyze(df,feature)
        
        
if __name__ == "__main__":
    ## Load the data
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    ## Analyzing a numerical feature
    analyzer = UnivariateAnalyzer(NumericalUnivariateAnalysis())
    analyzer.execute_strategy(df,"SalePrice")
    ## Analyzing a categorical feature
    analyzer.set_strategy(CategoricalUnivariateAnalysis())
    analyzer.execute_strategy(df , "Neighborhood")