import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.logger import info
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor_category.pkl')

class DataTransformationCategory:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            text_columns = 'full_text'

            numerical_columns = ["Customer Age"]

            categorical_columns = ["Ticket Priority","Ticket Channel", "Product Purchased", "Ticket Status", "Customer Gender"]

            text_pipeline = Pipeline(
                steps=[
                    ("tfidf", TfidfVectorizer(stop_words='english', max_features=5000))
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                    ("text_pipeline", text_pipeline, text_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            info("Error in get_data_transformer_object")
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            info("Read train and test data completed")

            train_df['full_text'] = train_df['Ticket Subject'].fillna('') + ' ' + train_df['Ticket Description'].fillna('')
            test_df['full_text'] = test_df['Ticket Subject'].fillna('') + ' ' + test_df['Ticket Description'].fillna('')

            info("Created full_text feature by combining Ticket Subject and Ticket Description")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "Ticket Type"
            drop_columns = [target_column_name,
                            'Ticket ID', 'Customer Name', 'Customer Email', 'Date of Purchase', 'Ticket Subject', 'Ticket Description', 'Resolution', 'First Response Time', 'Time to Resolution']

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)

            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)

            target_feature_test_df = test_df[target_column_name]

            le = LabelEncoder()
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            target_feature_test_arr = le.transform(target_feature_test_df)

            save_object(
                file_path=os.path.join('artifacts', 'label_encoder_category.pkl'),
                obj=le
            )

            info("Label encoding completed")
            info("Preprocessing")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)


            if hasattr(input_feature_train_arr, "toarray"):
                input_feature_train_arr = input_feature_train_arr.toarray()

            if hasattr(input_feature_test_arr, "toarray"):
                input_feature_test_arr = input_feature_test_arr.toarray()    


            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                obj=preprocessor_obj
            )
            info("Saved preprocessor object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path,
            )
        except Exception as e:
            info("Error in initiate_data_transformation")
            raise CustomException(e, sys)



if __name__ == "__main__":
    from src.components.data_ingestion_category import DataIngestionCategory

    data_ingestion = DataIngestionCategory()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformationCategory()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data_path, test_data_path
    )
    print(f"Train Array Shape: {train_arr.shape}")
    print(f"Test Array Shape: {test_arr.shape}")

