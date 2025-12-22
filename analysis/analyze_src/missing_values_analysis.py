from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 

class MissingValueAnalysisTemplate(ABC):
    def analyze(self,df:pd.DataFrame):
        self.identify_missing_values(df)
        self.visualize_missing_values(df)
    
    @abstractmethod
    def identify_missing_values(self,df:pd.DataFrame):
        pass
    
    def visualize_missing_values(self,df:pd.DataFrame):
        pass
    
    
class SimpleMissingValuesAnalysis(MissingValueAnalysisTemplate):
    def identify_missing_values(self, df:pd.DataFrame):
        print("Missing values count by column: ")
        missing_values = df.isnull().sum()
        print(missing_values)
    
    def visualize_missing_values(self, df:pd.DataFrame):
        print("Visualizing Missing Values: ")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.isnull(),cbar=False,cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
        

if __name__ == "__main__":
    
    #Load the data
    df = pd.read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    
    #Perform Missing Values Analysis
    missing_values_analyzer = SimpleMissingValuesAnalysis()
    
    missing_values_analyzer.analyze(df)