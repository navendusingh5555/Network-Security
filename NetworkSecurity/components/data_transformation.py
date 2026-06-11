import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from NetworkSecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS

from NetworkSecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from NetworkSecurity.entity.config_entity import DataTransformationConfig

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.utils.main_utils.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact : DataValidationArtifact = data_validation_artifact
            self.data_transformation_config : DataTransformationConfig = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    


    def get_data_transformer_object(cls) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
            cls: DataTransformation
        
        Returns:
            A Pipeline object
        """

        logging.info(f"Entered get_data_transformer_object method of DataTransformation class.")

        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initialize KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}")

            processor:Pipeline = Pipeline([("imputer", imputer)])
            return processor
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info(f"Entered initiate_data_transformation method of DataTransformation class.")

        try:
            logging.info(f"Starting data transformation.")

            ## Reading train and test data

            logging.info(f"Reading train and test data from validation artifact")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## Training DataFrame

            logging.info(f"Splitting train data into target feature and input feature")
            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN])
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

            ## Test DataFrame
            logging.info(f"Splitting test data into target feature and input feature")
            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN])
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)
            logging.info(f"Splitting of DataFrame completed successfully.")

            ## Applying Transformation over train and test DataFrame
            logging.info(f"Entered Transformation")
            preprocessor =self.get_data_transformer_object()

            logging.info(f"Applying fit_transform")
            preprocessor_object = preprocessor.fit(input_feature_train_df)
            transformed_input_train_df = preprocessor_object.transform(input_feature_train_df)

            transformed_input_test_df = preprocessor_object.transform(input_feature_test_df)

            ## Concatinating transformed input feature with target feature.
            logging.info(f"Concatinating transformed input feature with target feature")
            train_arr = np.c_[transformed_input_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_df, np.array(target_feature_test_df)]

            ## Save numpy array data
            logging.info(f"Saving Final Train, Test array and Preprocessor object.")
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_arr,)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_arr,)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            # Model Pusher
            save_object("final_model/preprocessor.pkl",preprocessor_object,)

            ## Preparing Artifacts
            logging.info(f"Preparing Artifacts.")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact
        
            logging.info(f"Artifacts: Output of DataTransformation Component, is created.")



        except Exception as e:
            raise NetworkSecurityException(e, sys)