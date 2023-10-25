import numpy as np
import pandas as pd
import os , sys
from sklearn.linear_model import LinearRegression , Ridge, Lasso , ElasticNet
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor
from src.utils import evaluate_model , save_obj

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainingConfig()

    def initiate_model_training(self,train_array,test_array):
        try :
            logging.info("Splitting  into dependentand and independent")
            x_train,x_test,y_train,y_test = (
                train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1]
            )
            models = {
                "Linear Regression" : LinearRegression(),
                "Ridge" : Ridge(),
                "Lasso" : Lasso(),
                "Elastic Net" : ElasticNet(),
                "Decision Tree" : DecisionTreeRegressor()
            }
            model_report : dict = evaluate_model(x_train,x_test,y_train,y_test,models)
            print(model_report)
            print("\n======================================\n")
            logging.info(f"Model Report : {model_report}")

            highest_r2_score = max(list(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(highest_r2_score)]

            best_model = models[best_model_name]
            print(f"Best Model Name : {best_model_name} Best model Score : {highest_r2_score}")
            print("\n======================================\n")
            logging.info(f"Best Model Name : {best_model_name} Best model Score : {highest_r2_score}")

            
            save_obj(
                file_path=self.model_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Error occoured in initiate model training")
            raise CustomException(e,sys)