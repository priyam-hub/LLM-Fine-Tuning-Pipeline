# DEPENDENCIES

import time
import torch
from tqdm import tqdm
from datetime import datetime

from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer

from ..utils.logger import LoggerSetup

llm_inference_logger = LoggerSetup(logger_name = "llm_inference.py", log_filename_prefix = "llm_inference").get_logger()

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
            
            model             {PreTrainedModel}          : The fine-tuned transformer model for text generation.
            
            tokenizer      {PreTrainedTokenizer}         : The tokenizer corresponding to the model.
            
            device             {str, optional}           : The device to run inference on ("cpu" or "cuda"). Defaults to "cpu".
        
        Raises:
            
            ValueError: If the model or tokenizer is not provided.
        """

        try:

            if model is None or tokenizer is None:
                llm_inference_logger.error("Model or tokenizer is not provided.")
                
                raise ValueError("Both model and tokenizer must be provided.")
            
            else:
                llm_inference_logger.info("Model and tokenizer loaded successfully.")
            
            self.model      = model
            self.tokenizer  = tokenizer
            self.device     = device
            self.model.to(self.device)

        except Exception as e:
            llm_inference_logger.error(f"Error initializing InferenceEngine: %s", {repr(e)})
            
            raise e

    
    def inference(self, prompt : str, max_length : int = 512, temperature : int = 0.7, num_return_sequences : int = 1) -> list:
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

        try:
        
            if self.model is None or self.tokenizer is None:
                llm_inference_logger.error("Model or tokenizer is not loaded.")
                
                raise ValueError("Model and tokenizer must be loaded first")
            
            else:
                llm_inference_logger.info("Model and tokenizer are loaded.")
            
            llm_inference_logger.info(f"Running inference with prompt: {prompt[:200]}...")
            
            self.model.to(self.device)

            inputs           = self.tokenizer(prompt, return_tensors = "pt").to(self.device)

            start_time       = time.time()
            start_dt         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            llm_inference_logger.info(f"[START] Inference started at: {start_dt}")

            generated_texts  = []
            
            with torch.no_grad():
                
                for _ in tqdm(range(num_return_sequences), desc = "Generating Text", unit = "seq"):
                
                    outputs  = self.model.generate(inputs.input_ids,
                                                max_length             = max_length,
                                                temperature            = temperature,
                                                num_return_sequences   = num_return_sequences,
                                                do_sample              = temperature > 0,
                                                pad_token_id           = self.tokenizer.eos_token_id
                                                )
                    
                    decoded  = self.tokenizer.decode(outputs[0], skip_special_tokens = True)
                    
                    generated_texts.append(decoded)
            
                    # generated_texts  = [self.tokenizer.decode(output, skip_special_tokens = True) for output in outputs]
            
            end_time         = time.time()
            end_dt           = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            duration         = end_time - start_time

            llm_inference_logger.info(f"\n[END] Inference completed at: {end_dt}")
            llm_inference_logger.info(f"[SUMMARY] Total time taken: {duration:.2f} seconds")
            llm_inference_logger.info(f"[SUMMARY] Number of sequences generated: {num_return_sequences}")

            return generated_texts
        
        except Exception as e:
            llm_inference_logger.error(f"Error during inference: %s", {repr(e)})
            
            raise e