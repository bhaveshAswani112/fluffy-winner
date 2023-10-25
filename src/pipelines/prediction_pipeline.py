import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.utils import load_obj


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try :
            preprocessor_path = os.path.join("artifacts",'preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_obj(preprocessor_path)
            model = load_obj(model_path)

            data_scaled = preprocessor.transform(features)
            predict = model.predict(data_scaled)
            return predict
    
        except Exception as e:
            logging.info("Error occoured during prediction of new data")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,carat:float,cut:str,color:str,clarity:str,depth:float,table:float,x:float,y:float,z:float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try :
            dic = {
                "carat" : [self.carat],
                "cut" : [self.cut],
                "color" : [self.color],
                "clarity" : [self.clarity],
                "depth" : [self.depth],
                "table" : [self.table],
                "x" : [self.x],
                "y" : [self.y],
                "z" : [self.z]

            }
            data = pd.DataFrame(dic)
            logging.info("DataFrame Gathered")
            return data
        
        except Exception as e:
            logging.info("Error occoured during dataframe formation")
            raise CustomException(e,sys)





