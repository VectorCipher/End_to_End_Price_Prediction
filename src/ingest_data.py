from abc import ABC,abstractmethod
import pandas as pd
import zipfile
import os


class DataIngestor(ABC):
    @abstractmethod ## decorator that forces child classes to implement a method 
    ## if not it will give an error
    def ingest(self,file_path:str) -> pd.DataFrame:
        pass


    
class ZipDataIngestor(DataIngestor):
    def ingest(self,file_path:str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file")
## Extract zipfile using zipfile in python 
        with zipfile.ZipFile(file_path,"r") as f:
            f.extractall("extracted_data")
## check whether there is a csv file or not 
        extracted_file = os.listdir("extracted_data")## filenames in the form of list 
        csv_files = [ f for f in extracted_file if f.endswith(".csv")]
        if len(csv_files) == 0:
            raise FileNotFoundError("No csv file is present in the extracted data.")
        if len(csv_files) > 1:
            raise ValueError("Multiple CSV File. Please specify which one to use.")
## If csv present then we have to convert it into DataFrame
        csv_file_path = os.path.join("extracted_data",csv_files[0])
        df = pd.read_csv(csv_file_path)
        
## Return the DataFrame
        return df 
    
    
    
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension:str) -> DataIngestor:
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension : {file_extension}")


if __name__ == "__main__":
    file_path = "C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/data/archive.zip"
    
    file_extension = os.path.splitext(file_path)[1]
    
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    
    df = data_ingestor.ingest(file_path)
    
    print(df.head())
        