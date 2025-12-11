import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.logger import info
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        info("Starting Data Ingestion")
        try:
            df = pd.read_csv("data/customer_support_tickets.csv")
            info("Read the dataset successfully")

            initial_cnt = df.shape[0]
            df = df.dropna(subset=['Customer Satisfaction Rating'])
            final_cnt = df.shape[0]

            info(f"Dropped rows with missing target values. Initial count: {initial_cnt}, Final count: {final_cnt}")

            info("Target to binary labels")
            df['Customer Satisfaction Rating'] = df['Customer Satisfaction Rating'].apply(lambda x: 1 if x >= 4 else 0)


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Customer Satisfaction Rating'])

            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            info("Data Ingestion completed successfully")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            info("Error occurred during Data Ingestion")
            raise CustomException(e, sys)
if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    obj = DataIngestion()
    obj.initiate_data_ingestion()
