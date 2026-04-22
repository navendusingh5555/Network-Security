from NetworkSecurity.components.data_ingestion import DataIngestion
from NetworkSecurity.components.data_validation import DataValidation
from NetworkSecurity.components.data_transformation import DataTransformation
from NetworkSecurity.components.model_trainer import ModelTrainer

from NetworkSecurity.exception.exception import NetworkSecurityException
from NetworkSecurity.logging.logger import logging

from NetworkSecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from NetworkSecurity.entity.config_entity import TrainingPipelineConfig

import sys


if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()

        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("Data Initiation Completed")
        print(dataingestionartifact)

        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(dataingestionartifact,data_validation_config)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("Data Validation Completed")
        print(data_validation_artifact)

        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        logging.info(f"Initializes Data Transformation.")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(f"Data Transformation Completed.")
        print(data_transformation_artifact)

        logging.info(f"Model Training Started.")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(model_trainer_config = model_trainer_config, data_transformation_artifact = data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info(f"Model Training Artifacts Created.")

    except Exception as e:
        raise NetworkSecurityException(e, sys)