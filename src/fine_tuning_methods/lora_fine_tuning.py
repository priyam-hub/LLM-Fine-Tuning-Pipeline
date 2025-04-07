# DEPENDENCIES

from tqdm import tqdm
from datetime import datetime

import transformers
from transformers import AutoModelForCausalLM

from peft import TaskType 
from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training

from ..utils.logger import LoggerSetup

lora_FT_logger = LoggerSetup(logger_name = "lora_fine_tuning.py", log_filename_prefix = "lora_fine_tuning").get_logger()


class LoRAFineTuning:
    """
    A class to handle LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning of language models.
    
    LoRA reduces the number of trainable parameters by applying low-rank adaptations
    to specific layers, making fine-tuning more memory efficient.
    """
    
    def __init__(self, model: transformers.PreTrainedModel) -> None:
        """
        Initializes the LoRAFineTuning class.
        
        Arguments:
        
            `model`             {transformers.PreTrainedModel}        : The pre-trained model to apply LoRA to.
        
        """
        try:

            self.model = model
            lora_FT_logger.info("LoRAFineTuning class initialized successfully.")

        except Exception as e:
            lora_FT_logger.error(f"Error initializing LoRAFineTuning class: {str(e)}")
            
            raise e


    def apply_lora(self, 
                   rank             : int    = 8, 
                   lora_alpha       : int    = 16, 
                   lora_dropout     : float  = 0.5, 
                   target_modules   : list   = None
                   ) -> transformers.PreTrainedModel:
        """
            Apply LoRA (Low-Rank Adaptation) to the model for parameter-efficient fine-tuning.

            LoRA reduces the number of trainable parameters by applying low-rank adaptations
            to specific layers, making fine-tuning more memory efficient.

            Arguments:

                `r`                         {int}             : Rank of the LoRA update matrices. A smaller rank reduces memory usage.
                
                `lora_alpha`                {int}             : Scaling factor for LoRA updates.
                
                `target_modules`       {list, optional}       : List of module names to apply LoRA to. 

            Raises:
                
                ValueError: If the model is not loaded before applying LoRA.

            Returns:
                
                `model` (transformers.PreTrainedModel): The model with LoRA applied.

            Functionality:

                - Detects the model type and selects appropriate layers for LoRA adaptation.
                - Supports different architectures like LLaMA, GPT, BERT, and T5.
                - Handles quantized models by preparing them for k-bit training.
                - Configures and applies LoRA using PEFT (Parameter-Efficient Fine-Tuning).
                - Prints the number of trainable parameters after applying LoRA.

            Example Usage:
                ```
                model = trainer.apply_lora(rank = 8, lora_alpha = 16, lora_droput = 0.5, target_modules = None)
                ```
        """

        try:

            if self.model is None:
                lora_FT_logger.error("Model must be loaded before applying LoRA.")
            
                raise ValueError("Model must be loaded first")
            
            else:
                lora_FT_logger.info("Model loaded successfully.")
            
            total_start_time       = datetime.now()
            
            lora_FT_logger.info(f"\nLoRA Application Started at: {total_start_time.strftime('%H:%M:%S')}\n")
            
            lora_FT_logger.info("Applying LoRA for parameter-efficient fine-tuning...")
            
            # Auto-detect target modules if not specified
            if target_modules is None:
                model_type         = self.model.config.model_type if hasattr(self.model, "config") else "unknown"
                
                lora_FT_logger.info(f"Auto-detecting target modules for model type: {model_type}")
                
                if "llama" in model_type.lower() or "mistral" in model_type.lower() or "gemma" in model_type.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                    
                    lora_FT_logger.info("Using LLaMA target modules: q_proj, v_proj, k_proj, o_proj")
                
                elif "gpt" in model_type.lower() or "falcon" in model_type.lower():
                    target_modules = ["c_attn", "c_proj"]
                    
                    lora_FT_logger.info("Using GPT target modules: c_attn, c_proj")
                
                elif "bert" in model_type.lower():
                    target_modules = ["query", "value", "key"]
                    
                    lora_FT_logger.info("Using BERT target modules: query, value, key")
                
                elif "t5" in model_type.lower():
                    target_modules = ["q", "v", "k", "o"]
                    
                    lora_FT_logger.info("Using T5 target modules: q, v, k, o")
                
                elif "bloom" in model_type.lower():
                    target_modules = ["query_key_value"]
                    
                    lora_FT_logger.info("Using BLOOM target modules: query_key_value")
                
                else:
                    lora_FT_logger.info(f"Auto-detection of target modules not supported for {model_type}. Using default q,v,k,o projections.")
                
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            

            if getattr(self.model, "is_quantized", False):
                self.model         = prepare_model_for_kbit_training(self.model)

                lora_FT_logger.info("Model is quantized. Preparing for k-bit training...")
            
            task_type              = TaskType.CAUSAL_LM if isinstance(self.model, AutoModelForCausalLM) else TaskType.SEQ_2_SEQ_LM
            
            lora_config            = LoraConfig(r               = rank,
                                                lora_alpha      = lora_alpha,
                                                target_modules  = target_modules,
                                                lora_dropout    = lora_dropout,
                                                bias            = "none",
                                                task_type       = task_type
                                                )

            lora_FT_logger.info("Configuring LoRA...")

            with tqdm(total       = 100, 
                    desc        = "Applying LoRA", 
                    bar_format  = "{l_bar}{bar} | {n_fmt}/{total_fmt} [Time Left: {remaining}]"
                    ) as pbar:
                
                model          = get_peft_model(self.model, lora_config)
                pbar.update(100)

            for name, param in self.model.named_parameters():
                
                if "lora" in name:
                
                    param.requires_grad = True 
                    
            model.print_trainable_parameters()

            self.model         = model

            total_end_time     = datetime.now()
            total_time_taken   = total_end_time - total_start_time

            lora_FT_logger.info("\nLoRA Applied Successfully!")
            lora_FT_logger.info(f"Started at    : {total_start_time.strftime('%H:%M:%S')}")
            lora_FT_logger.info(f"Finished at   : {total_end_time.strftime('%H:%M:%S')}")
            lora_FT_logger.info(f"Total Time Taken: {str(total_time_taken)}\n")

            return self.model
        
        except Exception as e:
            lora_FT_logger.error(f"Error applying LoRA: {repr(e)}")
            
            raise e