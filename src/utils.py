import pickle as pk
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
import os , sys
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score


def save_obj(file_path,obj):
    try :

        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as f:
            pk.dump(obj,f)

    except Exception as e :
        raise CustomException(e,sys)
    


def evaluate_model(x_train,x_test,y_train,y_test,models):
    try :
        report = {}
        for i in range(len(models.values())):
            model = list(models.values())[i]
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            score = r2_score(y_test,y_pred)
            mae = mean_absolute_error(y_test,y_pred)
            rmse = np.sqrt(mean_squared_error(y_test,y_pred))
            report[list(models.keys())[i]] = score

        return report
    

    except Exception as e:
        logging.info("Error occoured during evaluation of model")
        raise CustomException(e,sys)

def load_obj(path):
    try :
        with open(path,'rb') as f:
            return pk.load(f)
    except Exception as e:
        logging.info("Error occoured during loading pickle file")
        raise CustomException(e,sys)
