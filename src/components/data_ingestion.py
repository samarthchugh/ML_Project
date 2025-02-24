## DATA ingestion refers to process of collecting ,importing and sorting data from different sources into a system for further processing and analysis .
import os
import sys
from src.exception import CustomException # Custom exception handling
from src.logger import logging # logging module to track execution steps

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #used to simplify configuration class definition

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion paths.
    It defines the paths where raw, train, and test data will be stored.
    """
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    
class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestionConfig instance to access file paths.
        """
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Read data, performs train-test split, and saves the datasets.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading dataset
            df=pd.read_csv('notebook\data\Stud.csv')
            logging.info('Read the dataset as dataframe')
            
            # Creating the output directory if it does not exist
            # We use train_data_path here because all files reside in 'artifacts' folder
            # We can use any path here as we are only going to get directory name from them as they all are store in same dir
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # Saving the raw dataset before splitting
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            # Splitting data into training and testing sets
            logging.info("Train Test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            # Saving the split datasets
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of the data is completed.")
            
            # Return the train-test data paths for further transformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys) # Raising a custom exception for better error tracking
            
if __name__=="__main__":
    # Creating an instance of DataIngestion class
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)