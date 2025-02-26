import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

# Importing utility functions to save objects and evaluate models
from src.utils import save_object,evaluate_model

# Dataclass to store the file path where the trained model will be saved
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl') # Path to save the trained model

# ModelTrainer class to train and evaluate different regression models    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig() # Initialize the configuration for model training
        
    # Method to train and evaluate different regression models
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split Training and Test input data")
            # Splitting the training and test data into features(X) and target(y) variables
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],train_array[:,-1], # Features and target for training data 
                test_array[:,:-1],test_array[:,-1] # Features and target for test data
            )
            
            # Defining dctionary of different machine learning models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False), # Suppressing Catboost output
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
                
            # Evaluating models and storing their R2 scores in a dictionary
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                                models=models,param=params)
                
            # Finding the best model based on R2 score
            best_model_score=max(sorted(model_report.values()))
                
            # Identifying the model name corresponding to the best model score
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
                
            # Retreiving the best model based on the best model name
            best_model = models[best_model_name]
                
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
                
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
                
            predicted=best_model.predict(X_test)
            R2_score=r2_score(y_test,predicted)
            return R2_score
            
        except Exception as e:
            logging.error(f"Error in model training: {e}")
            raise CustomException(e,sys)
                