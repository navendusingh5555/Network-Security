from NetworkSecurity.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from NetworkSecurity.entity.config_entity import DataValidationConfig
from NetworkSecurity.exception.exception import NetworkSecurityException 
from NetworkSecurity.logging.logger import logging 
from NetworkSecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from NetworkSecurity.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config)
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def validate_number_of_numerical_columns(self, dataframe : pd.DataFrame) -> bool:
        try:
            number_of_num_columns = len(self._schema_config["columns"]["numerical_columns"])
            logging.info(f"Required number of numerical columns : {number_of_num_columns}")
            logging.info(
                f"Number of numerical columns in DataFrame : {len(dataframe.select_dtypes(include = ['number']).columns)}")
            
            if len(dataframe.select_dtypes(include = ['number']).columns) == number_of_num_columns:
                return True
            else:
                return False

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1,d2)

                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False

                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            """
            After we update the report, we also need to make sure that we need to get the report file path 
            so that we can put it over there.
            """

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            ## Create Directory for this :

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)

            """
            Now the one step that is remaining, i need to also go ahead and write the entire report 
            inside this particular file.
            """

            write_yaml_file(file_path = drift_report_file_path,content=report)

        except Exception as e:
            raise NetworkSecurityException(e,sys)
        
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_data_frame = DataValidation.read_data(train_file_path)
            test_data_frame = DataValidation.read_data(test_file_path)
            
            ## validate number of columns

            status = self.validate_number_of_columns(dataframe = train_data_frame)
            if not status:
                error_message=f"Train dataframe does not contain all columns.\n"
            status = self.validate_number_of_columns(dataframe = test_data_frame)
            if not status:
                error_message=f"Test dataframe does not contain all columns.\n"   

            ## lets check datadrift
            status = self.detect_dataset_drift(base_df = train_data_frame, current_df = test_data_frame)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok = True)

            train_data_frame.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_data_frame.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path = self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path = self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path = None,
                invalid_test_file_path = None,
                drift_report_file_path = self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e,sys)







    
    


    
                
            

            

            
            

            

            


        


    
    
    