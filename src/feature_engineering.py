from abc import ABC ,abstractmethod
import logging
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler,OneHotEncoder,MinMaxScaler

logging.basicConfig(level=logging.INFO, format =" %(asctime)s - %(levelname)s - %(message)s ",force=True )

## Abstract class for implementation of apply_transformation function
class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self,df:pd.DataFrame) -> pd.DataFrame:
        pass


## Lof transformation is a feature engineering technique used to 
## reduce skewness and compress large values by applying a log function.
class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features = features
    def apply_transformation(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Applying log transformation to features : {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed["feature"] = np.log1p(df[feature])
        logging.info("Log transformation completed.")
        return df_transformed
    
## Standard scaler is a feature scaling technique that standardizes 
## numerical data so that mean = 0 and S.D. = 1.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features =features
        self.scaler = StandardScaler()
    
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        logging.info(f"Applying standard scaling to features:{self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard Scaling Completed.")
        return df_transformed
    
## Min max Scaling is a feature scaling technique that rescales numerical values into 
## a fixed range usually 0 to 1.
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self,features,feature_range=(0,1)):
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def apply_transformation(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Applying Min - Max scaling to features :{self.features}")
        df_transformed = df.copy()
        df_transformed[self.features]=self.scaler.fit_transform(df[self.features])
        logging.info("Min - Max scaling completed.")
        return df_transformed
    
## This strategy applies one hot encoding to categorical features,converting them into binary vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self,features):
        self.features = features
        self.encoder = OneHotEncoder(sparse=False,drop="first")
        
    def apply_transformation(self, df:pd.DataFrame)-> pd.DataFrame:
        logging.info(f"Applying one hot encoding to features:{self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed=df_transformed.drop(columns=self.features)
        df_transformed = pd.concat([df_transformed,encoded_df],axis =1)
        logging.info("One hot encoding Completed.")
        return df_transformed
        
class FeatureEngineer:
    def __init__(self,strategy:FeatureEngineeringStrategy):
        self.strategy =strategy
        
    def set_strategy(self,startegy:FeatureEngineeringStrategy):
        logging.info("Switching feature engineering startegy")
        self.strategy = startegy
    def apply_feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        logging.info("Applying feature engineering startegy.")
        return self.strategy.apply_transformation(df)
    
if __name__ == "__main__":
    ##Load the dataframe
    df = pd .read_csv("C:/Users/Naitik/OneDrive/ドキュメント/Projects/End_to_End_Price_Prediction/src/extracted_data/AmesHousing.csv")
    # Log Transformation Example
    log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    df_log_transformed = log_transformer.apply_feature_engineering(df)

    #Standard Scaling Example
    standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

