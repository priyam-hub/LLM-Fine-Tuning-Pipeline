# DEPENDENCIES

import torch

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

class InferenceEngine:
    """
    A class for generating text using a fine-tuned transformer model.

    This class facilitates inference by generating text sequences based on a given prompt.
    It supports configuration of generation parameters such as `max_length`, `temperature`, and `num_return_sequences`.
    """
    
    def __init__(self, model : PreTrainedModel, tokenizer : PreTrainedTokenizer, device : str = "cpu") -> None:
        """
        Initializes the InferenceEngine with a fine-tuned model and tokenizer.

        Arguments:
            model      {PreTrainedModel}    : The fine-tuned transformer model for text generation.
            tokenizer  {PreTrainedTokenizer}: The tokenizer corresponding to the model.
            device     {str, optional}      : The device to run inference on ("cpu" or "cuda"). Defaults to "cpu".
        
        Raises:
            ValueError: If the model or tokenizer is not provided.
        """
        if model is None or tokenizer is None:
            raise ValueError("Both model and tokenizer must be provided.")
        
        self.model      = model
        self.tokenizer  = tokenizer
        self.device     = device
        self.model.to(self.device)

    
    def inference(self, prompt : str, max_length : int = 100, temperature : int = 0.7, num_return_sequences : int = 1) -> list:
        """
        Generate text using the fine-tuned model.

        Arguments:
            
            `prompt`                      {str}              : The input prompt to generate text from.
            
            `max_length`              {int, optional}        : The maximum length of the generated text. Defaults to 100.
            
            `temperature`            {float, optional}       : Sampling temperature for randomness (higher values increase randomness). Defaults to 0.7.
            
            `num_return_sequences`    {int, optional}        : The number of generated sequences to return. Defaults to 1.

        Returns:
            
            list: A list of generated text sequences.

        Raises:
            
            ValueError: If the model or tokenizer is not loaded.
        
        """
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        print(f"Running inference with prompt: {prompt[:50]}...")
        
        self.model.to(self.device)
        

        inputs           = self.tokenizer(prompt, return_tensors = "pt").to(self.device)
        
        with torch.no_grad():
            outputs      = self.model.generate(inputs.input_ids,
                                               max_length             = max_length,
                                               temperature            = temperature,
                                               num_return_sequences   = num_return_sequences,
                                               do_sample              = temperature > 0,
                                               pad_token_id           = self.tokenizer.eos_token_id
                                               )
        
        generated_texts  = [self.tokenizer.decode(output, skip_special_tokens = True) for output in outputs]
        
        return generated_texts