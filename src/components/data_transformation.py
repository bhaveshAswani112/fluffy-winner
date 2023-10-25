from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Handling missing values
from sklearn.preprocessing import OrdinalEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import os ,sys
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts",'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    

    def get_data_transformation_object(self):
        try :
            logging.info("Data Transformation started")
            cat_col = ["cut","color","clarity"]
            num_col = ["carat","depth","table","x","y","z"]

            cut_cat = ["Fair"  , "Good"  , "Very Good"  , "Premium"  , "Ideal" ]
            color_cat = ["D"  , "E"  , "F"  , "G"  , "H" ,"I","J"]
            clar_cat = ["I1"  , "SI2"  , "SI1"  , "VS2"  , "VS1" ,"VVS2","VVS1","IF"]

            logging.info("pipeline Initiated")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OrdinalEncoder(categories=[cut_cat,color_cat,clar_cat])),
                    ("scaler",StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_col),
                ("cat_pipeline",cat_pipeline,cat_col)
            ]
            )
            logging.info("Data Transformation completed")
            return preprocessor
        
        except Exception as e :
            logging.info("Error Occoured in transformation")
            raise CustomException(e,sys)
        


    def initiate_data_transformation(self,train_path,test_path):
        try :
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info("training and test data loaded")
            logging.info(f"training data head {train_data.head().to_string()}")
            logging.info(f"test data head {test_data.head().to_string()}")

            logging.info("Preprocessing starts")

            preprocessor = self.get_data_transformation_object()
            
            drop = ["price","id"]
            target = ["price"]

            input_feature_train = train_data.drop(drop , axis =1)
            target_feature_train = train_data[target]


            input_feature_test = test_data.drop(drop , axis=1)
            target_feature_test = test_data[target]

            logging.info("Split into dependent and independent feature")

            # Transformation
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor.transform(input_feature_test)

            logging.info("Applying preprocessing object on  training and test datasets")


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test)]

            logging.info("Combining preprocessed data with independent feature")

            save_obj(file_path=self.transformation_config.preprocessor_obj_path,obj=preprocessor)

            logging.info("Preprocessor pickle is created and saved")

            return(train_arr,test_arr,self.transformation_config.preprocessor_obj_path)
        
        except Exception as e :
            logging.info("An error occoured during data transformation")
            raise CustomException(e,sys) 


