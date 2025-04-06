# DEPENDENCIES

import torch
import datasets
from tqdm import tqdm
from datetime import datetime

from trl import SFTTrainer

import transformers
from transformers import Trainer
from transformers import PreTrainedTokenizer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling

from ..fine_tuning_methods.lora_fine_tuning import LoRAFineTuning

class SupervisedFineTuning:
    """
    A class to handle supervised fine-tuning (SFT) of a language model using the SFTTrainer from TRL.
    """
    
    def __init__(self, 
                 model       : AutoModelForCausalLM, 
                 tokenizer   : PreTrainedTokenizer,
                 dataset     : datasets.DatasetDict, 
                 device      : str = "cuda" if torch.cuda.is_available() else "cpu"
                 ) -> None:
        """
        Initializes the SupervisedFineTuning class.

        Arguments:
        
            `model`         {AutoModelForCausalLM}      : The pre-trained language model to fine-tune.
            
            `tokenizer`      {PreTrainedTokenizer}      : The tokenizer associated with the model.
            
            `dataset`             {DatasetDict}         : The dataset prepared for fine-tuning.
            
            `device`             {str, optional}        : The device to run training on ('cuda' or 'cpu'). Defaults to GPU if available.
        
        """

        lora_adapter            = LoRAFineTuning(model = model)
    
        quantized_model         = lora_adapter.apply_lora(rank            = 8, 
                                                          lora_alpha      = 16, 
                                                          lora_dropout    = 0.1,
                                                          target_modules  = ["query", "key", "value"]
                                                          )
        
        self.model              = quantized_model
        self.device             = device
        self.tokenizer          = tokenizer
        self.prepared_dataset   = dataset

        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = True

    def apply_supervised_fine_tuning(self, 
                                     output_dir       : str   = "./supervised_fine_tuned_model", 
                                     batch_size       : int   = 8, 
                                     learning_rate    : float = 2e-5, 
                                     num_epochs       : int   = 3
                                     ) -> AutoModelForCausalLM:
        """
        Fine-tunes the model using supervised fine-tuning (SFT) with the SFTTrainer from TRL.

        Arguments:

            output_dir        {str, optional}      : Directory to save the fine-tuned model. Defaults to "./sft_model".
            
            batch_size        {int, optional}      : Batch size for training. Defaults to 8.
            
            learning_rate    {float, optional}     : Learning rate for training. Defaults to 2e-5.
            
            num_epochs        {int, optional}      : Number of training epochs. Defaults to 3.

        Raises:

            ValueError  : If the model or prepared dataset is not loaded.

        Returns:

            model       : The fine-tuned model.

        Functionality:

            - Configures training arguments for supervised fine-tuning.
            - Uses `SFTTrainer` from TRL for training with packing enabled.
            - Applies gradient accumulation to handle larger batch sizes effectively.
            - Supports mixed precision training (`fp16`) if running on CUDA.
            - Saves the fine-tuned model and tokenizer to the specified output directory.
        
        """
        if self.model is None or self.prepared_dataset is None:
            raise ValueError("Model and prepared dataset must be loaded first")
        
        self.prepared_dataset["train"] = self.prepared_dataset["train"].with_format("torch")
        
        print("Starting supervised fine-tuning...")
        
        training_args      = TrainingArguments(output_dir                   = output_dir,
                                               per_device_train_batch_size  = batch_size,
                                               gradient_accumulation_steps  = 4,
                                               learning_rate                = learning_rate,
                                               num_train_epochs             = num_epochs,
                                               save_strategy                = "epoch",
                                               save_total_limit             = 2,
                                               logging_dir                  = f"{output_dir}/logs",
                                               logging_steps                = 10,
                                               fp16                         = False,
                                               remove_unused_columns        = False,
                                               report_to                    = "none",
                                               optim                        = "adamw_torch",
                                               lr_scheduler_type            = "cosine",
                                               warmup_steps                 = 0,
                                               )
        
        trainer            = SFTTrainer(model            = self.model,
                                        args             = training_args,
                                        train_dataset    = self.prepared_dataset["train"],
                                        eval_dataset     = self.prepared_dataset["test"],
                                        data_collator    = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, 
                                                                                           mlm       = False),
                                        )
        
        total_start_time   = datetime.now()
        
        print(f"Fine-tuning started at: {total_start_time.strftime('%H:%M:%S')}")
        
        print("Training model...")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = datetime.now()
        
            print(f"\nEpoch {epoch}/{num_epochs} started at {epoch_start_time.strftime('%H:%M:%S')}")
            
            with tqdm(total       = len(self.prepared_dataset["train"]) // batch_size, 
                      desc        = f"Epoch {epoch}/{num_epochs}", 
                      bar_format  = "{l_bar}{bar} | {n_fmt}/{total_fmt} [Time Left: {remaining}]", 
                      ncols       = 80
                      ) as pbar:
                
                trainer.train()
                
                pbar.update(pbar.total)
            
            epoch_end_time = datetime.now()
            time_taken     = epoch_end_time - epoch_start_time
            
            print(f"Epoch {epoch} finished at {epoch_end_time.strftime('%H:%M:%S')} | Time Taken: {str(time_taken)}")
        
        total_end_time     = datetime.now()
        total_time_taken   = total_end_time - total_start_time
        
        print("\nSupervised Fine-tuning Completed!")
        print(f"Total Training Time: {str(total_time_taken)}")
        print(f"Started at: {total_start_time.strftime('%H:%M:%S')}")
        print(f"Finished at: {total_end_time.strftime('%H:%M:%S')}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to: {output_dir}")
        
        return self.model, self.tokenizer
