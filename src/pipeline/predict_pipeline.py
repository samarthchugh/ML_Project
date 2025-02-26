import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            logging.info("Entered the prediction method")
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/preprocessor.pkl'
            model=load_object(file_path=model_path)   # Load the model object
            preprocessor=load_object(preprocessor_path) # Load the preprocessor object
            data_scaled=preprocessor.transform(features) # Transform the input features
            preds=model.predict(data_scaled) # Predict the target variable
            logging.info("Prediction completed")
            return preds
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise CustomException(e,sys)
    
    
class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethinicty:str,
                 parental_level_of_education:str,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score,
                 writing_score):
        self.gender = gender
        self.race_ethinicty=race_ethinicty
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race/ethnicity":[self.race_ethinicty],
                "parental level of education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test preparation course":[self.test_preparation_course],
                "reading score":[self.reading_score],
                "writing score":[self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.error(f"Error in converting data to dataframe: {e}")
            raise CustomException(e,sys)