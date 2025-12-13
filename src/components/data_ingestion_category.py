import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import info
from src.exception import CustomException

@dataclass
class DataIngestionConfig:
  raw_data_path: str = os.path.join('artifacts', 'category_data.csv')
  train_data_path: str = os.path.join('artifacts', 'train_category.csv')
  test_data_path: str = os.path.join('artifacts', 'test_category.csv')

class DataIngestionCategory:
  def __init__(self):
    self.ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self):
    info("Starting data ingestion")

    try:
      df = pd.read_csv('data/customer_support_tickets.csv') 
      info("Data read successfully from CSV")

      initial_cnt = df.shape[0]

      df = df.dropna(subset=['Ticket Type'])
      final_cnt = df.shape[0]
      info(f"Dropped {initial_cnt - final_cnt} rows with missing 'Ticket Type'")

      os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) 

      df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

      train_set, test_set = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Ticket Type'])

      train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

      test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

      info("Data ingestion completed successfully")

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )
    except Exception as e:
      info("Error occurred during data ingestion")
      raise CustomException(e, sys)
    
if __name__ == "__main__":
  obj = DataIngestionCategory()
  obj.initiate_data_ingestion()    
  