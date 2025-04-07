# DEPENDENCIES

import datasets

import transformers
from transformers import PreTrainedTokenizer

from ..utils.logger import LoggerSetup
from ..fine_tuning_methods.lora_fine_tuning import LoRAFineTuning
from ..fine_tuning_methods.supervised_fine_tuning import SupervisedFineTuning
from ..fine_tuning_methods.instruction_fine_tuning import InstructionFineTuning

llm_fineTuning_logger = LoggerSetup(logger_name = "llm_fine_tuning.py", log_filename_prefix = "llm_fine_tuning").get_logger()

class LLMFineTuner:
    """
    A class for fine-tuning transformer-based models using different methods.

    This class supports supervised fine-tuning, instruction-based adaptation, and reinforcement learning from human feedback (RLHF). Additionally, it allows for parameter-efficient fine-tuning using LoRA.
    """
    
    def __init__(self, model : transformers.PreTrainedModel, tokenizer : PreTrainedTokenizer , prepared_dataset : datasets.DatasetDict) -> None:
        
        """
        Initializes the ModelFineTuner with a pre-trained model.

        Arguments:
        
            `model`               {transformers.PreTrainedModel}           : The pre-trained model to be fine-tuned.
            
            `tokenizer`                {PreTrainedTokenizer}               : The tokenizer associated with the model.
            
            `dataset`                       {DatasetDict}                  : The dataset prepared for fine-tuning.
        
        This instance will provide methods to apply different fine-tuning strategies, including LoRA-based tuning for efficiency.
        """

        try:
        
            self.model            = model
            self.tokenizer        = tokenizer
            self.prepared_dataset = prepared_dataset if prepared_dataset else {}

            llm_fineTuning_logger.info("LLMFineTuner initialized successfully")

        except Exception as e:
            llm_fineTuning_logger.error(f"Error initializing LLMFineTuner: {repr(e)}")
            
            raise e

    def fine_tune(self, method : str = "supervised", use_lora : bool = False, lora_config : dict = None, **kwargs) -> transformers.PreTrainedModel:
        
        """
        Fine-tune the model using the specified fine-tuning method.

        Arguments:
            
            method                {str}           : The fine-tuning method to use. Options are:
                
                - "supervised"                    : Supervised fine-tuning using labeled data.
                
                - "instruction"                   : Instruction-based fine-tuning for task-specific adaptation.
                
                - "rlhf"                          : Reinforcement Learning from Human Feedback (RLHF).
            
            use_lora              {bool}          : Whether to apply LoRA for parameter-efficient fine-tuning.
            
            lora_config      {dict, optional}     : Configuration parameters for LoRA if use_lora is True.
            
            **kwargs: Additional keyword arguments specific to the chosen fine-tuning method.

        Returns:
            
            transformers.PreTrainedModel: The fine-tuned model.

        Raises:
            
            ValueError: If an unsupported fine-tuning method is provided.
        
        """

        try:

            if use_lora:
                lora_params = lora_config or {}
                
                LoRAFineTuning.apply_lora(**lora_params)
                
                llm_fineTuning_logger.info("Applied LoRA for parameter-efficient fine-tuning")

            if method   == "supervised":
                llm_fineTuning_logger.info("Applying supervised fine-tuning")
                
                return SupervisedFineTuning.apply_supervised_fine_tuning(**kwargs)
            
            elif method == "instruction":
                llm_fineTuning_logger.info("Applying instruction-based fine-tuning")
                
                return InstructionFineTuning.apply_instruction_fine_tuning(**kwargs)
            
            elif method == "rlhf":
                llm_fineTuning_logger.info("Applying RLHF fine-tuning")
                
                return self.apply_rlhf(**kwargs)
            
            else:
                llm_fineTuning_logger.error(f"Unsupported fine-tuning method: {repr(method)}")
                
                raise ValueError(f"Unsupported fine-tuning method: {method}")
            
        except Exception as e:
            llm_fineTuning_logger.error(f"Error during fine-tuning: {repr(e)}")
            
            raise e