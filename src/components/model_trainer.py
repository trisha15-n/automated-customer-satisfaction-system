import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from src.exception import CustomException
from src.logger import info
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
  trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_array, test_array):
    try:
      info("Splitting the data in test and train data")
      #Last column is Target (Customer Satisfaction Rating) 
      X_train, y_train, X_test, y_test = (
          train_array[:, :-1], 
          train_array[:, -1], 
          test_array[:, :-1], 
          test_array[:, -1]
      )    

      y_train = y_train.astype(int)
      y_test = y_test.astype(int)

      unique_target = np.unique(y_train)
      info(f"Unique target classes: {unique_target}")

      if -1 in unique_target or -1.0 in unique_target:
        info("CRITICAL WARNING! Detected -1 model taking sentiment score as target")


      #For XGBoost
      #y_train = y_train - 1
      #y_test = y_test - 1


      models = {
        "Logistic Regression" : LogisticRegression(class_weight='balanced'),
        "Random Forest Classifier" : RandomForestClassifier(class_weight='balanced'),
        "XGBoost" : XGBClassifier(scale_pos_weight=1.55,use_label_encoder=False, eval_metric='logloss')
      }

      models_report = {}
      #models_report_p = {}
      models_report_r = {}
      for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        models_report[model_name] = accuracy

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        #models_report_p[model_name] = precision
        models_report_r[model_name] = recall


        info(f"{model_name} Accuracy: {accuracy}")
        info(f"{model_name} Precision: {precision}")
        info(f"{model_name} Recall: {recall}")

      best_model_score = max(models_report_r.values())

      best_model_name = list(models_report_r.keys())[list(models_report_r.values()).index(best_model_score)]
      
      best_model = models[best_model_name]



      """if best_model_accuracy < 0.6:
        info("No model found with accuracy greater than 60%")
      
      info(f"Best Model Found: {best_model_name} with accuracy {best_model_accuracy}")"""

      info(f"Best Model Found: {best_model_name} with recall {best_model_score}")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,  
        obj=best_model
      )

      predicted = best_model.predict(X_test)
      acc = accuracy_score(y_test, predicted)

      return acc
    except Exception as e:
      raise CustomException(e, sys)
    

if __name__== "__main__":
  from src.components.data_ingestion import DataIngestion
  from src.components.data_transformation import DataTransformation

  try:
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_array, test_array, _ = data_transformation.initiate_data_transformation(
      train_data_path,
      test_data_path
    )

    model_trainer = ModelTrainer()
    acc = model_trainer.initiate_model_trainer(train_array, test_array)
    print(f"Model Accuracy: {acc}")
  except Exception as e:
    raise CustomException(e, sys)
      
      