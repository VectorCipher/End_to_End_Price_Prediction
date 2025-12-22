from abc import ABC , abstractmethod
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd

class MultiVariateAnalysisTemplate(ABC):
    def analyze(self,df:pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)
    
    @abstractmethod
    def generate_correlation_heatmap(self,df:pd.DataFrame):
        pass
    @abstractmethod
    def generate_pairplot(self,df:pd.DataFrame):
        pass
    

class SimpleMultiVariateAnalysis(MultiVariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df:pd.DataFrame):
        plt.figure(figsize=(12,10))
        sns.heatmap(df.corr(),annot=True,fmt=".2f",cmap="coolwarm",linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()
    
    def generate_pairplot(self, df:pd.DataFrame):
        sns.pairplot(df)
        plt.suptitle("Pair plot of selected features",y=1.02)
        plt.show()
        
if __name__=="__main__":
    #Load the data
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    # Perform Multivariate Analysis
    multivariate_analyzer = SimpleMultiVariateAnalysis()

    # Select important features for pair plot
    selected_features = df[['SalePrice', 'Gr Liv Area', 'Overall Qual', 'Total Bsmt SF', 'Year Built']]

    # Execute the analysis
    multivariate_analyzer.analyze(selected_features)
