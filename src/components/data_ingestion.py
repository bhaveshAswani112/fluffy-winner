import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')


class DataIngestion :
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_config(self):
        logging.info("Data Ingestion Method Starts")

        try :
            data = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info("Dataset read as pandas dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            logging.info("Directory Created")
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Raw Data Stored")
            train_set , test_set = train_test_split(data , test_size=0.3 ,random_state=30)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False , header = True)

            logging.info("Data Ingestion Completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("Error Occoured in Data Ingestion Config")