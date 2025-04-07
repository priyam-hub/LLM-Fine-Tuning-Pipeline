import os
from pathlib import Path

class Config:
    
    """
    Configuration class for storing credentials, file paths, and URLs.
    It also provides a method to ensure required directories exist.
    """

    HUGGING_FACE_DATASET               = "imdb"
    HUGGING_FACE_MODEL_ID              = "google-bert/bert-base-uncased"

    REWARD_MODEL_PATH                  = "./model/reward_model"
    RLHF_FINE_TUNED_MODEL_PATH         = "./model/rlhf_fine_tuned_model"
    SUPERVISED_FINE_TUNED_MODEL_PATH   = "./model/supervised_fine_tuned_model"
    INSTRUCTION_FINE_TUNED_MODEL_PATH  = "./model/instruction_fine_tuned_model"

    @staticmethod
    def setup_directories():
        """
        Ensures that all required directories exist.
        If a directory does not exist, it creates it.
        """
        
        directories = []
        
        for directory in directories:
            
            if not os.path.exists(directory):
            
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            
            else:
                print(f"Directory already exists: {directory}")