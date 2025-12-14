import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from src.utils import save_object
from src.exception import CustomException
from src.logger import info


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model_category.pkl")

class ModelTrainerCategory:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            info("Splitting training and testing data")
            X_train, y_train, X_test, y_test =(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial'),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            }   

            models_report = {}

            for model_name, model in models.items():
                info(f"Training {model_name}")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                models_report[model_name] = accuracy

                info(f"{model_name} Accuracy: {accuracy}")


            best_model_score = max(models_report.values())
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            info(f"Best model found: {best_model_name} with accuracy {best_model_score}")
            info(f"\n Classification Report for {best_model_name}:\n{classification_report(y_test, best_model.predict(X_test))}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    #from src.components.data_ingestion_category import DataIngestionCategory
    from src.components.data_transformation_category import DataTransformationCategory


    try:
        #data_ingestion = DataIngestionCategory()
        #train_data, test_data = data_ingestion.initiate_data_ingestion()

        import os
        train_data = os.path.join("artifacts", "train_category.csv")
        test_data = os.path.join("artifacts", "test_category.csv")

        info("Using Tagged data")

        data_transformation = DataTransformationCategory()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data) 

        model_trainer = ModelTrainerCategory()
        best_model_score = model_trainer.initiate_model_trainer(train_array, test_array)
        info(f"Best Model Score: {best_model_score}")
    except Exception as e:
        raise CustomException(e, sys)
         


               
