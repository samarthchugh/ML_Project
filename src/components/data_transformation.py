import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# DataTransformationConfig stores the path where the preprocessor object (pickle file) will be saved
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
# Class responsible for transforming the data (handling missing values, scaling, encoding categorical variables, etc.)
class DataTransformation:
    def __init__(self):
        # Initialize the configuration for data transformation
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        Creates and returns a preprocessing pipeline for numerical and categorical features.
        '''
        
        try:
            # Define numerical and categorical columns
            numerical_columns = ['writing score','reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            
            # Define transformation pipelinefor numerical columns
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), # handle missing values using median imputation
                    ('scaler',StandardScaler()) # Scale numerical values to have zero mean and unit variance
                ]
            )
            
            # Define the transformation pipeline for categorical columns
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')), # Handle missing categorical values with most frequent value
                    ('one_hot_encoder',OneHotEncoder()) # Convert categorical values to one-hot encoded format
                ]
            )

            # Log the column names for debugging purposes
            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")
            
             # Combine both pipelines using ColumnTransformer
            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns), # Apply numerical pipeline to numerical columns
                    ('cat_pipeline',cat_pipeline,categorical_columns) # Apply categorical pipeline to categorical columns
                ]
            )
            
            return preprocessor # return preprocessing pipeline
            
        except Exception as e:
            logging.error(f"Error in {__name__}: {str(e)}", exc_info=True)
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        Reads train and test datasets, applies preprocessing transformations, 
        and returns transformed arrays along with the preprocessor object path.
        """
        try:
            # Load training and testing dataset from csv files
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            
            logging.info("Read Train-Test data completed")
            
            # Obtain the preprocessing pipeline
            logging.info("Obtaining pre-processing object")
            preprocessor_obj=self.get_data_transformer_object()
            
            # Define the target column (label to predict)
            target_column_name = 'math score'
            numerical_columns = ['writing score','reading score']
            
            # Separate input features (X) and target (y) for training data
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            # Separate input features (X) and target (y) for testing data
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            # Fit and transform training data using the preprocessor pipeline
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            # Transform test data using the pre-trained pipeline (without fitting again)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)
            
            # Combine transformed features with the target variable to form final train & test datasets
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            logging.info(f'Saving preprocessing object.')
            
            # Save the preprocessor object for future use (to ensure consistency during inference)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            # Return transformed train & test arrays along with the path to the preprocessor object
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error(f"Error in {__name__}: {str(e)}", exc_info=True)
            raise CustomException(e,sys)