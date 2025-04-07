# DEPENDENCIES

from pyexpat import model
from regex import E
import torch

import transformers
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM

from ..utils.logger import LoggerSetup

modelLoader_logger = LoggerSetup(logger_name = "model_loader.py", log_filename_prefix = "model_loader").get_logger()



class ModelLoader:
    """
    A class to handle downloading and loading of pretrained LLMs from Hugging Face.
    """
    
    def __init__(self):
        """
        Initializes the ModelLoader instance with model and tokenizer set to None.

        Attributes:

            model              : The language model to be fine-tuned (initially None)
            tokenizer          : The tokenizer associated with the model (initially None)

        """

        try:
        
            self.model                = None
            self.tokenizer            = None

            modelLoader_logger.info("ModelLoader initialized successfully")

        except Exception as e:
            modelLoader_logger.error(f"Error initializing ModelLoader: {repr(e)}")
            raise e

    def download_model(self, model_id : str) -> tuple:
        """
        Download a pretrained LLM from Hugging Face.
        
        Arguments:

            `model_id`        {str}      : The Hugging Face model ID

        Returns:

            Tuple [torch.nn.Module, transformers.PreTrainedTokenizer]: The loaded model and tokenizer.
        
        """

        try:
        
            modelLoader_logger.info(f"Downloading model {model_id}...")
            
            if any(size in model_id.lower() for size in ['7b', '13b', '70b', 'llama']):
                
                modelLoader_logger.info("Loading large model with quantization...")
                
                bnb_config               = BitsAndBytesConfig(load_in_4bit               = True,
                                                            bnb_4bit_use_double_quant  = True,
                                                            bnb_4bit_quant_type        = "nf4",
                                                            bnb_4bit_compute_dtype     = torch.bfloat16
                                                            )
                
                # Use AutoModelForCausalLM for decoder-only models
                self.model               = AutoModelForCausalLM.from_pretrained(model_id,
                                                                                quantization_config  = bnb_config,
                                                                                device_map           = "auto",
                                                                                trust_remote_code    = True
                                                                                )
            
            else:
                
                # Check if model is an encoder-decoder model or decoder-only
                if any(arch in model_id.lower() for arch in ['t5', 'bart', 't0']):
                    self.model           = AutoModelForSeq2SeqLM.from_pretrained(model_id)

                    modelLoader_logger.info("Loading encoder-decoder (Seq2SeqLM) model...")
                
                else:
                    self.model           = AutoModelForCausalLM.from_pretrained(model_id)

                    modelLoader_logger.info("Loading decoder-only (CausalLM) model...")
            
            self.tokenizer               = AutoTokenizer.from_pretrained(model_id)
            
            # Ensure padding token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

                modelLoader_logger.info("Padding token set to EOS token")
            
            else:
                modelLoader_logger.info("Padding token already exists")
                
            modelLoader_logger.info(f"Model {model_id} loaded successfully")
            
            return self.model, self.tokenizer
        
        except Exception as e:
            modelLoader_logger.error(f"Error downloading model {model_id}: {repr(e)}")
            raise e
