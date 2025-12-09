import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import info
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Customer Age']
            ordinal_columns = ['Ticket Priority']
            nominal_columns = ['Ticket Type', "Ticket Status", "Ticket Channel", 'Product Purchased', 'Customer Gender'] 

            priority_categories = ['Low', 'Medium', 'High', 'Critical']

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            ord_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[priority_categories])),
                ('scaler', StandardScaler())
            ])

            nom_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_columns),
                ('ord_pipeline', ord_pipeline, ordinal_columns),
                ('nom_pipeline', nom_pipeline, nominal_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            info("Read train and test data completed")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'Customer Satisfaction Rating'
            drop_columns = [target_column_name, 'Ticket ID', 'Customer Name',"Customer Email",'Date of Purchase','Ticket Subject' ,'Ticket Description', 'Resolution', 'First Response Time', "Time to Resolution"]

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)

            target_feature_test_df = test_df[target_column_name]

            info("Applying preprocessing object on training and testing dataframes.")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()    


            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]



            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj  
            )
            info("Preprocessor object saved successfully.")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

        info("Data Transformation completed successfully.")
        info(f"Train array shape: {train_arr.shape}")
        info(f"Test array shape: {test_arr.shape}")
    except Exception as e:
        info("Error occurred in the main execution")
        raise CustomException(e, sys)
    
        