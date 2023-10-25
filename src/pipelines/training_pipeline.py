from src.components.data_ingestion import DataIngestion
from src.logger import logging
from src.exception import CustomException
import os , sys
import pandas as pd
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    obj = DataIngestion()
    train_path , test_path = obj.initiate_config()
    print(train_path,test_path)

    transformation = DataTransformation()
    train_arr , test_arr  ,_ = transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)

    

