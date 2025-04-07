# DEPENDENCIES

import torch
import datasets
from tqdm import tqdm
from datetime import datetime

import transformers
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

from ..fine_tuning_methods.lora_fine_tuning import LoRAFineTuning

from ..utils.logger import LoggerSetup

instructionFT_logger = LoggerSetup(logger_name = "instruction_fine_tuning.py", log_filename_prefix = "instruction_fine_tuning").get_logger()


class InstructionFineTuning:
    """
    A class to handle instruction-based fine-tuning of a language model using the Hugging Face `Trainer` API.
    """
    
    def __init__(self, model : AutoModelForCausalLM, tokenizer : transformers.PreTrainedTokenizer, dataset : datasets.DatasetDict, device : str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        """
        Initializes the InstructionFineTuning class.    

        Arguments:

            `model`        {AutoModelForCausalLM}  : The pre-trained language model to fine-tune.
            
            `tokenizer`    {PreTrainedTokenizer}   : The tokenizer associated with the model.
            
            `dataset`      {DatasetDict}           : The dataset prepared for fine-tuning.
            
            `device`       {str, optional}        : The device to run training on ('cuda' or 'cpu'). Defaults to GPU if available.
        """
        
        try:

            lora_adapter      = LoRAFineTuning(model = model)
        
            quantized_model   = lora_adapter.apply_lora(rank            = 8, 
                                                        lora_alpha      = 16, 
                                                        lora_dropout    = 0.1,
                                                        target_modules  = ["query", "key", "value"]
                                                        )

            self.model                  = quantized_model
            self.tokenizer              = tokenizer
            self.prepared_dataset       = dataset
            self.device                 = device

            for param in self.model.parameters():
                param.requires_grad     = True

            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

            if torch.cuda.is_available() and torch.__version__ >= "2.0":
                self.model = torch.compile(self.model)

            self.model.train()

        except Exception as e:
            instructionFT_logger.exception(f"Error initializing InstructionFineTuning: {repr(e)}")
            raise

    def apply_instruction_fine_tuning(self, 
                                      output_dir     : str   = "./instruction_fine_tuned_model", 
                                      batch_size     : int   = 8, 
                                      learning_rate  : float = 5e-5, 
                                      num_epochs     : int   = 3
                                      ) -> AutoModelForCausalLM:
        """
        Fine-tunes the model using an instruction-based dataset.

        Arguments:

            `output_dir`          {str, optional}    : Path to save the fine-tuned model. Defaults to "./instruction_ft_model".
            
            `batch_size`          {int, optional}    : Batch size for training. Defaults to 8.
            
            `learning_rate`      {float, optional}   : Learning rate for training. Defaults to 5e-5.
            
            `num_epochs`          {int, optional}    : Number of training epochs. Defaults to 3.

        Raises:
            
            ValueError: If the model or prepared dataset is not loaded.

        Returns:
            
            model: The fine-tuned model.

        Functionality:
            
            - Configures training arguments for fine-tuning.
            - Uses the Hugging Face `Trainer` class for training.
            - Applies full fine-tuning (no parameter-efficient tuning).
            - Saves the fine-tuned model and tokenizer to the specified output directory.
        
        """
        
        if self.model is None or self.prepared_dataset is None:
            raise ValueError("Model and prepared dataset must be loaded first")
        
        mixed_precision   = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"

        print("Starting instruction fine-tuning...")
        
        training_args     = TrainingArguments(output_dir                   = output_dir,
                                              per_device_train_batch_size  = batch_size,
                                              learning_rate                = learning_rate,
                                              num_train_epochs             = num_epochs,
                                              save_strategy                = "steps",
                                              save_steps                   = 500,
                                              save_total_limit             = 2,
                                              logging_dir                  = f"{output_dir}/logs",
                                              logging_steps                = 50,
                                              fp16                         = (mixed_precision == "fp16"),
                                              bf16                         = (mixed_precision == "bf16"),
                                              gradient_accumulation_steps  = 4,  
                                              dataloader_num_workers       = 4, 
                                              report_to                    = "none"  
                                              )
        
        trainer           = Trainer(model           = self.model,
                                    args            = training_args,
                                    train_dataset   = self.prepared_dataset["train"],
                                    data_collator   = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, 
                                                                                      mlm       = False
                                                                                      ),
                                    compute_metrics = None,
                                    callbacks       = None,
                                    )
         
        total_start_time     = datetime.now()
        
        print(f"Fine-tuning started at: {total_start_time.strftime('%H:%M:%S')}")

        self.model.train()
        
        print("Training model...")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = datetime.now()
            
            print(f"\nEpoch {epoch}/{num_epochs} started at {epoch_start_time.strftime('%H:%M:%S')}")
            
            with tqdm(total        = len(self.prepared_dataset["train"]) // batch_size, 
                      desc         = f"Epoch {epoch}/{num_epochs}", 
                      bar_format   = "{l_bar}{bar} | {n_fmt}/{total_fmt} [Time Left: {remaining}]", 
                      ncols        = 80
                      ) as pbar:
            
                trainer.train()
            
                pbar.update(pbar.total)
            
            epoch_end_time   = datetime.now()
            
            time_taken       = epoch_end_time - epoch_start_time
            
            print(f"Epoch {epoch} finished at {epoch_end_time.strftime('%H:%M:%S')} | Time Taken: {str(time_taken)}")
        
        total_end_time       = datetime.now()
        total_time_taken     = total_end_time - total_start_time
        
        print("\nFine-tuning Completed!")
        print(f"Total Training Time: {str(total_time_taken)}")
        print(f"Started at: {total_start_time.strftime('%H:%M:%S')}")
        print(f"Finished at: {total_end_time.strftime('%H:%M:%S')}")
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to: {output_dir}")
        
        return self.model, self.tokenizer