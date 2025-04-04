# DEPENDENCIES

import torch

import transformers
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM


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
        
        self.model                = None
        self.tokenizer            = None

    def download_model(self, model_id : str) -> tuple:
        """
        Download a pretrained LLM from Hugging Face.
        
        Arguments:

            `model_id`        {str}      : The Hugging Face model ID

        Returns:

            Tuple [torch.nn.Module, transformers.PreTrainedTokenizer]: The loaded model and tokenizer.
        
        """
        
        print(f"Downloading model {model_id}...")
        
        if any(size in model_id.lower() for size in ['7b', '13b', '70b', 'llama']):
            
            print("Loading large model with quantization...")
            
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
            
            else:
                self.model           = AutoModelForCausalLM.from_pretrained(model_id)
        
        self.tokenizer               = AutoTokenizer.from_pretrained(model_id)
        
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model {model_id} loaded successfully")
        
        return self.model, self.tokenizer
