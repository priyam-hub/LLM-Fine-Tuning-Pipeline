# DEPENDENCIES

from tqdm import tqdm
from datetime import datetime

import transformers
from transformers import AutoModelForCausalLM

from peft import TaskType 
from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training

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
        self.model = model

    def apply_lora(self, rank : int = 8, lora_alpha : int = 16, lora_dropout : float = 0.5, target_modules : list = None) -> transformers.PreTrainedModel:
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
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        total_start_time       = datetime.now()
        print(f"\nLoRA Application Started at: {total_start_time.strftime('%H:%M:%S')}\n")
        
        print("Applying LoRA for parameter-efficient fine-tuning...")
        
        # Auto-detect target modules if not specified
        if target_modules is None:
            model_type         = self.model.config.model_type if hasattr(self.model, "config") else "unknown"
            
            if "llama" in model_type.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            elif "gpt" in model_type.lower():
                target_modules = ["c_attn", "c_proj"]
            
            elif "bert" in model_type.lower():
                target_modules = ["query", "value", "key"]
            
            elif "t5" in model_type.lower():
                target_modules = ["q", "v", "k", "o"]
            
            else:
                print(f"Auto-detection of target modules not supported for {model_type}. Using default q,v,k,o projections.")
                
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        

        if getattr(self.model, "is_quantized", False):
            self.model         = prepare_model_for_kbit_training(self.model)
        
        task_type              = TaskType.CAUSAL_LM if isinstance(self.model, AutoModelForCausalLM) else TaskType.SEQ_2_SEQ_LM
        
        lora_config            = LoraConfig(r               = rank,
                                            lora_alpha      = lora_alpha,
                                            target_modules  = target_modules,
                                            lora_dropout    = lora_dropout,
                                            bias            = "none",
                                            task_type       = "SEQ_CLS"
                                            )

        print("Configuring LoRA...")

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

        print("\nLoRA Applied Successfully!")
        print(f"Started at    : {total_start_time.strftime('%H:%M:%S')}")
        print(f"Finished at   : {total_end_time.strftime('%H:%M:%S')}")
        print(f"Total Time Taken: {str(total_time_taken)}\n")

        return self.model