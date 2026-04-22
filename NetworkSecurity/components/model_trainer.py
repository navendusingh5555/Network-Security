import os
import sys

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from NetworkSecurity.entity.config_entity import ModelTrainerConfig

from NetworkSecurity.utils.main_utils.utils import save_object, load_object
from NetworkSecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models

from NetworkSecurity.utils.ml_utils.model.estimator import NetworkModel
from NetworkSecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from xgboost import XGBClassifier


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    
    def train_model(self, X_train, y_train, X_test, y_test):
        models = {
            "Random Forest" : RandomForestClassifier(verbose = 1),
            "Decision Tree" : DecisionTreeClassifier(),
            "Logistic Regression" : LogisticRegression(verbose = 1),
            "Gradient Boosting" : GradientBoostingClassifier(verbose = 1),
            "Adaboost" : AdaBoostClassifier(),
            "Xgboost" : XGBClassifier(),
        }

        params = {
            "Random Forest" : {
                "criterion" : ["gini", "entropy", "log_loss"],
                "max_features" : ["sqrt", "log2", None],
                "n_estimators" : [8, 16, 32, 64, 128, 256]
            },

            "Decision Tree" : {
                "criterion" : ["gini", "entropy", "log_loss"],
                "splitter" : ["best", "random"],
                "max_features" : ["sqrt", "log2"]   
            },

            "Logistic Regression" : {},

            "Gradient Boosting" : {
                "loss" : ["log_loss", "exponential"],
                "learning_rate" : [0.1, 0.01, 0.05, 0.001],
                "subsample" : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "criterion" : ["friedman_mse", "squared_error"],
                "max_features" : ["sqrt", "log2"],
                "n_estimators" : [8, 16, 32, 64, 128, 256]
            },

            "Adaboost" : {
                "learning_rate" : [0.1, 0.01, 0.5, 0.001],
                "n_estimators" : [8, 16, 32, 64, 128, 256]
            },

            "Xgboost" : {
                "learning_rate" : [0.1, 0.01, 0.5, 0.001],
                "max_depth" : [5, 8, 12, 20, 30],
                "n_estimators" : [8, 16, 32, 64, 128, 256],
                "colsample_bytree" : [0.5, 0.8, 1, 0.3, 0.4]
            }
            
            
        }

        model_report : dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                              models = models, params = params)
        
        ## To get best model score from dict
        best_model_score = max(sorted(model_report.values()))

        ## To get best model name from dict
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

        best_model = models[best_model_name]
        y_train_pred = best_model.predict(X_train)
        classification_train_metric = get_classification_score(y_true = y_train, y_pred = y_train_pred)

        y_test_pred = best_model.predict(X_test)
        classification_test_metric = get_classification_score(y_true = y_test, y_pred = y_test_pred)


        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok = True)

        Network_Model = NetworkModel(preprocessor = preprocessor, model = best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj = NetworkModel)

        ## Model Trainer Artifact
        model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact = classification_train_metric,
                             test_metric_artifact = classification_test_metric
                             )
        
        logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")

        return model_trainer_artifact


    

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Reading data from Transformation Artifact.")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading train array and test array.")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info(f"Splitting into train and test.")
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)