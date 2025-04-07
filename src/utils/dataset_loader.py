# DEPENDENCIES

import os
import datasets
from datasets import DatasetDict
from datasets import load_dataset
from regex import E

from ..utils.logger import LoggerSetup

datasetLoader_logger = LoggerSetup(logger_name = "dataset_loader.py", log_filename_prefix = "<dataset_loader>").get_logger()


class DatasetLoader:
    """
    A class to handle loading datasets for fine-tuning from either a local file or the Hugging Face Hub.
    """
    
    def __init__(self):
        """
        Initializes the DatasetLoader instance with dataset set to None.
        
        Attributes:
            
            dataset            : The dataset to be used for fine-tuning (initially None)

        """

        try:
        
            self.dataset = None
            datasetLoader_logger.info("DatasetLoader initialized successfully")
        
        except Exception as e:
            datasetLoader_logger.error(f"Error initializing DatasetLoader: {repr(e)}")


    def load_dataset(self, dataset : str) -> datasets.DatasetDict:
            
        """
        Loads a dataset for fine-tuning from either a local file or the Hugging Face Hub.

        This function first attempts to load the dataset from the Hugging Face `datasets` library.  
        If that fails, it checks if the dataset exists as a local file and loads it based on its file extension.

        Supported file formats:
        - `.csv`  : Loaded using `datasets.load_dataset('csv')`
        - `.json` : Loaded using `datasets.load_dataset('json')`
        - `.txt`  : Reads line-by-line and stores as a list under the 'train' key

        Arguments:

            `dataset`            {str}         : Either the name of a Hugging Face dataset (e.g., `'imdb'`)  
                                                 or a local file path (e.g., `'data/train.csv'`).

        Returns:
            
            datasets.DatasetDict               : The loaded dataset with a 'train' split.

        Raises:
            
            ValueError                         : If the dataset is not found or the file format is unsupported.
        
        """
    
        print(f"Loading dataset from {dataset}...")
        
        # LOADING DATASET FROM HUGGING FACE HUB
        try:

            self.dataset = load_dataset(dataset)
            
            datasetLoader_logger.info(f"Loaded dataset from Hugging Face: {dataset}")
        
        # LOADING DATASET FROM LOCAL FILE
        except Exception as e:

            if os.path.exists(dataset):
                extension = os.path.splitext(dataset)[1]
                
                if extension == '.csv':
                    self.dataset = load_dataset('csv', data_files = dataset)
            
                elif extension == '.json':
                    self.dataset = load_dataset('json', data_files = dataset)
            
                elif extension == '.txt':
                    with open(dataset, 'r') as f:
                        texts    = [line.strip() for line in f]
                    self.dataset = {'train': texts}
            
                else:
                    raise ValueError(f"Unsupported file extension: {extension}")
                    
                datasetLoader_logger.info(f"Loaded dataset from local file: {dataset}")
        
            else:
                datasetLoader_logger.error(f"Dataset {dataset} not found")
                raise ValueError(f"Dataset {dataset} not found")
            
            datasetLoader_logger.error(f"Error loading dataset: {repr(e)}")
        
        print(f"Dataset loaded with {len(self.dataset['train'])} training examples")
        
        return self.dataset
    
    def save_dataset(self, dataset : DatasetDict, file_path : str) -> None:
        """
        Saves the loaded dataset to a specified file path in CSV, JSON, or TXT format.

        Arguments:

            dataset          {DatasetDict}     : The dataset to be saved.
        
            file_path            {str}         : The destination file path with extension (.csv, .json, or .txt).

        Raises:
            
            ValueError : If the file extension is unsupported.
        
        """
        try:
        
            extension        = os.path.splitext(file_path)[1]
            
            if extension    == '.csv':
                dataset['train'].to_csv(file_path, index=False)
            
            elif extension  == '.json':
                dataset['train'].to_json(file_path, orient = 'records', lines = True)
            
            elif extension  == '.txt':
                
                with open(file_path, 'w') as f:
                
                    for item in dataset['train']:
                
                        f.write(f"{item}\n")
            
            else:
                datasetLoader_logger.error(f"Unsupported file extension: {extension}")
                raise ValueError(f"Unsupported file extension: {extension}")
            
            datasetLoader_logger.info(f"Dataset saved successfully to {file_path}")

        except Exception as e:
            datasetLoader_logger.error(f"Error saving dataset: {repr(e)}")
            
            raise ValueError(f"Error saving dataset: {repr(e)}")
