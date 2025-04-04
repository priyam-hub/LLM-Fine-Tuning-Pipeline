# DEPENDENCIES

import os
import time
import torch
import datasets
import evaluate
import numpy as np
from tqdm import tqdm
from typing import Optional
from datetime import datetime

from datasets import load_dataset

from nltk.translate.bleu_score import corpus_bleu

from peft import TaskType
from peft import LoraConfig
from peft import get_peft_model
from peft import prepare_model_for_kbit_training

from trl import PPOConfig
from trl import SFTTrainer
from trl import PPOTrainer
from trl import create_reference_model
from trl import AutoModelForCausalLMWithValueHead

import transformers
from transformers import Trainer
from transformers import pipeline
from transformers import PPOConfig
from transformers import PPOTrainer
from transformers import AutoTokenizer
from transformers import TrainerCallback
from transformers import PreTrainedModel
from transformers import TrainingArguments
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSeq2SeqLM
from transformers import TextGenerationPipeline
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLMWithValueHead


class LLMFineTuner:

    """
    A comprehensive class for fine-tuning large language models using various techniques.
    
    This class provides an end-to-end solution for downloading pre-trained language models,
    loading and preparing datasets, applying different fine-tuning methods (including 
    instruction tuning, supervised fine-tuning, and RLHF), and evaluating model performance.
    
    The LLMFineTuner supports parameter-efficient fine-tuning with LoRA, prompting techniques
    like Chain-of-Thought, and evaluation metrics including perplexity, coherence, and BLEU scores.
    
    Features:

        - Download models from Hugging Face Hub
        - Load datasets from local files or Hugging Face datasets
        - Prepare datasets for various fine-tuning approaches
        - Apply different fine-tuning methods (instruction, supervised, RLHF)
        - Use parameter-efficient techniques like LoRA
        - Enhance reasoning with Chain-of-Thought prompting
        - Run inference with fine-tuned models
        - Evaluate models with standard metrics
    
    Attributes:

        model              : The language model to be fine-tuned
        tokenizer          : The tokenizer associated with the model
        dataset            : The raw dataset for fine-tuning
        prepared_dataset   : The processed dataset ready for fine-tuning
        device             : The device to use for training (CUDA or CPU)
    
    Example:

        >>> fine_tuner = LLMFineTuner()
        >>> fine_tuner.download_model("gpt2")
        >>> fine_tuner.load_dataset("squad")
        >>> fine_tuner.prepare_dataset(instruction_column = "question", response_column = "answer")
        >>> fine_tuner.apply_lora()
        >>> fine_tuner.fine_tune(method = "instruction", output_dir = "./my_model")
        >>> outputs    = fine_tuner.inference("What is machine learning?")
        >>> metrics    = fine_tuner.evaluate(metric="all")
    
    """
    
    def __init__(self) -> None:

        """
        Initialize the LLMFineTuner class for fine-tuning large language models.
        
        This class provides a comprehensive toolkit for downloading, preparing, fine-tuning,
        and evaluating large language models using various techniques including instruction
        fine-tuning, supervised fine-tuning, RLHF, Chain-of-Thought prompting, and LoRA.
        
        Attributes:

            model              : The language model to be fine-tuned (initially None)
            tokenizer          : The tokenizer associated with the model (initially None)
            dataset            : The raw dataset for fine-tuning (initially None)
            prepared_dataset   : The processed dataset ready for fine-tuning (initially None)
            device             : The device to use for training ("cuda" if available, otherwise "cpu")

        """
         
        self.model             = None
        self.tokenizer         = None
        self.dataset           = None
        self.prepared_dataset  = None
        self.device            = "cuda" if torch.cuda.is_available() else "cpu"
        
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
    
    def load_dataset(self, dataset : str) -> datasets.Datasetdict:
        
        """
        Loads a dataset for fine-tuning from either a local file or the Hugging Face Hub.

        This function first attempts to load the dataset from the Hugging Face `datasets` library.  
        If that fails, it checks if the dataset exists as a local file and loads it based on its file extension.

        Supported file formats:
        - `.csv`  : Loaded using `datasets.load_dataset('csv')`
        - `.json` : Loaded using `datasets.load_dataset('json')`
        - `.txt`  : Reads line-by-line and stores as a list under the 'train' key

        Arguments:

            `dataset`            {str}         : Either the name of a Hugging Face dataset (e.g., `'imdb'`)  
                                                 or a local file path (e.g., `'data/train.csv'`).

        Returns:
            
            datasets.DatasetDict               : The loaded dataset with a 'train' split.

        Raises:
            
            ValueError                         : If the dataset is not found or the file format is unsupported.
        
        """
       
        print(f"Loading dataset from {dataset}...")
        
        # LOADING DATASET FROM HUGGING FACE HUB
        try:

            self.dataset = load_dataset(dataset)
            
            print(f"Loaded dataset from Hugging Face: {dataset}")
        
        # LOADING DATASET FROM LOCAL FILE
        except:

            if os.path.exists(dataset):
                extension = os.path.splitext(dataset)[1]
                
                if extension == '.csv':
                    self.dataset = load_dataset('csv', data_files = dataset)
            
                elif extension == '.json':
                    self.dataset = load_dataset('json', data_files = dataset)
            
                elif extension == '.txt':
                    with open(dataset, 'r') as f:
                        texts    = [line.strip() for line in f]
                    self.dataset = {'train': texts}
            
                else:
                    raise ValueError(f"Unsupported file extension: {extension}")
                    
                print(f"Loaded dataset from local file: {dataset}")
           
            else:
                raise ValueError(f"Dataset {dataset} not found")
        
        print(f"Dataset loaded with {len(self.dataset['train'])} training examples")
        
        return self.dataset
    
    def prepare_dataset(self, 
                        max_length          : int = 512, 
                        instruction_column  : str = None, 
                        response_column     : str = None, 
                        text_column         : str = None, 
                        prompt_template     : str = None
                        ) -> datasets.DatasetDict:
        
        """
        Prepares the dataset for fine-tuning by tokenizing and formatting input-output pairs.

        This function supports two modes of dataset preparation:
        
        1. **Instruction Fine-Tuning**: If `instruction_column` and `response_column` are provided,  
             the dataset is formatted with input-output pairs based on the `prompt_template`.
        
        2. **Regular Language Model Fine-Tuning**: If `text_column` is provided, the dataset is tokenized  
             as a standard language modeling dataset.

        Arguments:

            `max_length`               {int, optional}          : Maximum sequence length for tokenization. Defaults to 512.
            
            `instruction_column`       {str, optional}          : Column name containing instructions (for instruction tuning).
            
            `response_column`          {str, optional}          : Column name containing responses (for instruction tuning).
            
            `text_column`              {str, optional}          : Column name containing plain text (for standard LM fine-tuning).
            
            `prompt_template`          {str, optional}          : Template for formatting prompts  
        

        Returns:
        
            datasets.DatasetDict: The tokenized dataset ready for fine-tuning.

        Raises:
            
            ValueError: If the tokenizer or dataset is not loaded before calling this function.
        """
        
        if self.tokenizer is None or self.dataset is None:
            raise ValueError("Model tokenizer and dataset must be loaded first")
        
        print("Preparing dataset for fine-tuning...")
        
        # FOR INSTRUCTION FINE-TUNING
        
        if instruction_column and response_column:
        
            def format_instruction(example : dict) -> dict:
                """
                Formats an example for instruction fine-tuning by generating input-output text pairs.

                Arguments:

                    example         {dict}      : A dictionary representing a single data instance,  
                                                  containing at least `instruction_column` and `response_column`.

                Returns:
                
                    dict: A modified example with two new keys:
                        - `"input_text"`: The formatted instruction prompt.
                        - `"output_text"`: The corresponding response.

                Behavior:
                    - If `prompt_template` is provided, it formats the instruction using the given template.
                    - If no template is provided, it defaults to `"Instruction: {instruction}\nResponse: "`.
                """
                
                if prompt_template:
                    prompt                 = prompt_template.format(instruction = example[instruction_column])
                    example["input_text"]  = prompt
                    example["output_text"] = example[response_column]
        
                else:
                    example["input_text"]  = f"Instruction: {example[instruction_column]}\nResponse: "
                    example["output_text"] = example[response_column]
        
                return example
            
            tokenized_dataset              = self.dataset.map(format_instruction)
            
            def tokenize_function(examples : dict) -> dict:
                """
                Tokenizes input and output text for fine-tuning language models.

                Arguments:

                    examples                 {dict}           : A batch of dataset examples containing:
                        - `"input_text"`      {str}           : The input text (e.g., an instruction prompt).
                        - `"output_text"` {str, optional}     : The expected response text (for encoder-decoder models).

                Returns:

                    dict: A dictionary containing tokenized inputs with the following keys:
                        - `"input_ids"`: Tokenized input sequences.
                        - `"attention_mask"`: Attention masks for padding.
                        - `"labels"`: Tokenized output sequences (for supervised fine-tuning).

                Behavior:

                    - **For decoder-only models (`AutoModelForCausalLM`)**: 
                    - The `labels` are set to the same value as `input_ids` (self-supervised learning).
                    - **For encoder-decoder models**:
                    - The `input_text` is tokenized separately for input.
                    - The `output_text` is tokenized to create the `labels`.

                """
        
                model_inputs               = self.tokenizer(examples["input_text"], 
                                                            truncation = True, 
                                                            max_length = max_length, 
                                                            padding    = "max_length",
                                                            )
                
                # FOR DECODER-ONLY MODELS
                if isinstance(self.model, AutoModelForCausalLM):
                    model_inputs["labels"] = model_inputs["input_ids"].copy()
        
                # FOR ENCODER-ONLY MODELS
                else:
                    
                    labels                 = self.tokenizer(examples["output_text"],
                                                            truncation   = True,
                                                            max_length   = max_length,
                                                            padding      = "max_length",
                                                            )
                    
                    model_inputs["labels"] = labels["input_ids"]
                    
                return model_inputs
        
        # FOR REGULAR LANGUAGE MODEL FINE-TUNING
        else:
            text_column                    = text_column or "text"
            
            def tokenize_function(examples : str) -> dict:
                """
                Tokenizes input text for fine-tuning a language model.

                Arguments:

                    examples      {str}     : A batch of dataset examples containing the text to be tokenized.

                Returns:

                    dict: A dictionary containing the tokenized output with the following keys:
                        - `"input_ids"`     : Tokenized input sequences.
                        - `"attention_mask"`: Attention masks indicating which tokens are padding.

                Behavior:
                
                    - Tokenizes the text from the specified `text_column`.
                    - Truncates sequences longer than `max_length`.
                    - Pads shorter sequences to `max_length`.

                """
                
                return self.tokenizer(examples[text_column],
                                      truncation   = True,
                                      max_length   = max_length,
                                      padding      = "max_length",
                                      )
        
        print("Tokenizing dataset...")
        
        tokenized_dataset                   = self.dataset.map(tokenize_function,
                                                               batched        = True,
                                                               remove_columns = self.dataset["train"].column_names)
        
        self.prepared_dataset               = tokenized_dataset
        
        print("Dataset preparation completed")
        
        return self.prepared_dataset
    
    def apply_instruction_fine_tuning(self, 
                                      output_dir     : str   = "./instruction_ft_model", 
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
        
        print("Starting instruction fine-tuning...")
        
        training_args     = TrainingArguments(output_dir                   = output_dir,
                                              per_device_train_batch_size  = batch_size,
                                              learning_rate                = learning_rate,
                                              num_train_epochs             = num_epochs,
                                              save_strategy                = "epoch",
                                              save_total_limit             = 2,
                                              logging_dir                  = f"{output_dir}/logs",
                                              logging_steps                = 10,
                                              fp16                         = True if self.device == "cuda" else False,
                                              )
        
        trainer           = Trainer(model           = self.model,
                                    args            = training_args,
                                    train_dataset   = self.prepared_dataset["train"],
                                    tokenizer       = self.tokenizer,
                                    data_collator   = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, 
                                                                                      mlm       = False
                                                                                      ),
                                    compute_metrics = None,
                                    callbacks       = None,
                                    )
         
        total_start_time     = datetime.now()
        
        print(f"Fine-tuning started at: {total_start_time.strftime('%H:%M:%S')}")
        
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
        
        return self.model
    
    def apply_supervised_fine_tuning(self, 
                                     output_dir       : str   = "./sft_model", 
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
                                               fp16                         = True if self.device == "cuda" else False,
                                               remove_unused_columns        = False,
                                               report_to                    = "none",
                                               optim                        = "paged_adamw_32bit",
                                               lr_scheduler_type            = "cosine",
                                               warmup_steps                 = 0,
                                               )
        
        trainer            = SFTTrainer(model            = self.model,
                                        args             = training_args,
                                        train_dataset    = self.prepared_dataset["train"],
                                        tokenizer        = self.tokenizer,
                                        packing          = True,
                                        max_seq_length   = 512,
                                        data_collator    = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, 
                                                                                           mlm       = False),
                                        )
        
        print("Training model...")
        trainer.train()
        
        # SAVE THE MODEL
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Supervised fine-tuning completed. Model saved to {output_dir}")
        
        return self.model
    
    def apply_rlhf(self, 
                   output_dir : str = "./rlhf_model", 
                   reward_model_id : str = None,
                   batch_size : int = 4, 
                   learning_rate : float = 1e-5, 
                   num_epochs : int = 1
                   ) -> AutoModelForCausalLM:
        
        """
        Fine-tunes the model using Reinforcement Learning from Human Feedback (RLHF) 
        with the Proximal Policy Optimization (PPO) algorithm.

        Arguments:

            output_dir           {str, optional}       : Directory to save the fine-tuned model. Defaults to "./rlhf_model".
            
            reward_model_id      {str, optional}       : Hugging Face model ID for the reward model.     
            
            batch_size           {int, optional}       : Batch size for PPO training. Defaults to 4.
            
            learning_rate       {float, optional}      : Learning rate for PPO. Defaults to 1e-5.
            
            num_epochs           {int, optional}       : Number of training epochs. Defaults to 1.

        Raises:
            
            ValueError: If the model or prepared dataset is not loaded.

        Returns:
            
            model: The fine-tuned model with a value head trained via RLHF.

        Functionality:

            - Initializes an RLHF-compatible model with a value head for reward learning.
            - Creates a reference model for KL divergence penalty in PPO training.
            - Loads a reward model from Hugging Face if `reward_model_id` is provided.
            - If no reward model is provided, uses a simple heuristic-based reward function.
            - Configures the PPO trainer for RLHF.
            - Generates responses using the fine-tuned model and evaluates them with the reward model.
            - Trains the model using PPO by optimizing responses based on computed rewards.
            - Saves the trained model and tokenizer to the specified output directory.

        Example Usage:

            ```
            model = trainer.apply_rlhf(output_dir="./rlhf_model", reward_model_id="OpenAI/reward-model")
            ```
        
        """
        
        if self.model is None or self.prepared_dataset is None:
            raise ValueError("Model and prepared dataset must be loaded first")
        
        print("Starting RLHF training...")
        
        model                = AutoModelForCausalLMWithValueHead.from_pretrained(self.model.config._name_or_path 
                                                                                 if hasattr(self.model, "config") 
                                                                                 else self.model.name_or_path,
                                                                                 trust_remote_code = True)

        ref_model            = create_reference_model(model)

        if reward_model_id:
            reward_model     = AutoModelForCausalLM.from_pretrained(reward_model_id)
            reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
            reward_pipeline  = pipeline("text-generation", model = reward_model, tokenizer = reward_tokenizer)
        
        else:
            
            print("Warning: No reward model provided. Using a simple length-based reward for demonstration.")
            
            def reward_fn(response : list) -> list:
                """
                Computes a simple heuristic-based reward for a given response based on its length.

                Arguments:

                    response       {list of str}     : A list of generated response texts.

                Returns:

                    list of float                    : A list of reward scores (between 0 and 1), where the score 
                                                       is calculated as the number of words in each response divided by 50, 
                                                       capped at a maximum of 1.

                Functionality:

                    - Counts the number of words in each response.
                    - Normalizes the count by dividing by 50.
                    - Caps the reward at 1.0 for responses with 50 or more words.

                """
                
                return [min(len(r.split()), 50) / 50.0 for r in response]
        
        ppo_config           = PPOConfig(batch_size      = batch_size,
                                         learning_rate   = learning_rate,
                                         ppo_epochs      = num_epochs,
                                         model_name      = model.config._name_or_path 
                                                           if hasattr(model, "config") 
                                                           else model.name_or_path,
                                         )
        
        ppo_trainer          = PPOTrainer(config       = ppo_config,
                                          model        = model,
                                          ref_model    = ref_model,
                                          tokenizer    = self.tokenizer,
                                          )

        gen_kwargs           = {"max_new_tokens": 100,"temperature": 0.7,"top_p": 0.9,}

        prompts              = [example["input_text"] for example in self.prepared_dataset["train"]]
        
        # Start RLHF training
        for epoch in range(num_epochs):
            
            print(f"RLHF Epoch {epoch+1}/{num_epochs}")
            
            for i, prompt in enumerate(tqdm(prompts[:100])):  # Limit for demonstration
                
                # Generate responses from the model
                response_tensors  = ppo_trainer.generate(prompt, **gen_kwargs)
                response_texts    = [self.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                

                if reward_model_id:
                    rewards       = [reward_pipeline(r)[0]["score"] for r in response_texts]
            
                else:
                    rewards       = reward_fn(response_texts)
                
                stats             = ppo_trainer.step(response_tensors, rewards)
                
                if i % 10 == 0:

                    print(f"Batch {i} stats: {stats}")
        
        # SAVE THE FINAL MODEL
        ppo_trainer.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.model                = model
        
        print(f"RLHF training completed. Model saved to {output_dir}")
        
        return self.model
    
    
    def apply_lora(self, r : int = 8, lora_alpha : int = 16, target_modules : list = None) -> transformers.PreTrainedModel:
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
                model = trainer.apply_lora(r=8, lora_alpha=16)
                ```
        """
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
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
        
        lora_config            = LoraConfig(r               = r,
                                            lora_alpha      = lora_alpha,
                                            target_modules  = target_modules,
                                            lora_dropout    = 0.05,
                                            bias            = "none",
                                            task_type       = task_type
                                            )

        model                  = get_peft_model(self.model, lora_config)
        
        model.print_trainable_parameters()
        
        self.model             = model
        
        print("LoRA applied successfully")
        
        return self.model
    
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
        if use_lora:
            lora_params = lora_config or {}
            self.apply_lora(**lora_params)
            
            print("Applied LoRA for parameter-efficient fine-tuning")

        if method   == "supervised":
            return self.apply_supervised_fine_tuning(**kwargs)
        
        elif method == "instruction":
            return self.apply_instruction_fine_tuning(**kwargs)
        
        elif method == "rlhf":
            return self.apply_rlhf(**kwargs)
        
        else:
            raise ValueError(f"Unsupported fine-tuning method: {method}")
    
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
    
    def evaluate(self, test_dataset = None, metric : str ="perplexity") -> dict:
        """
        Evaluate the fine-tuned model using specified evaluation metrics.

        Arguments:
            
            `test_dataset`    {optional}       : The dataset to evaluate the model on. If None, uses `self.prepared_dataset["test"]`
                                               if available; otherwise, defaults to `self.prepared_dataset["train"]`.
            
            `metric`        {str, optional}    : The evaluation metric to compute. Choices are:
                                                 - "perplexity": Measures how well the model predicts the test dataset.
                                                 - "coherence": Assesses the likelihood of next tokens given previous context.
                                                 - "bleu": Evaluates text similarity using BLEU score.
                                                 - "all": Computes all available metrics.
                                                 Defaults to "perplexity".

        Returns:

            dict: A dictionary containing evaluation results for the specified metrics.

        Raises:

            ValueError: If the model or tokenizer is not loaded before evaluation.

        Notes:
        
            - Perplexity is computed using the average loss over token sequences.
            - Coherence is measured using the log probability of the actual next tokens.
            - BLEU score compares generated outputs with reference texts.
            - The evaluation may use a subset of the dataset (e.g., first 100 examples for coherence and BLEU).
        """
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded first")
        
        dataset                    = test_dataset or self.prepared_dataset["test"] if "test" in self.prepared_dataset else self.prepared_dataset["train"]
        
        print(f"Evaluating model with {metric} metric...")
        
        results                    = {}
        
        self.model.to(self.device)
        
        if metric == "perplexity" or metric == "all":
            
            print("Calculating perplexity...")
            
            self.model.eval()
            total_loss             = 0
            total_tokens           = 0
            
            with torch.no_grad():
            
                for i in tqdm(range(0, len(dataset), 8)):  
            
                    batch          = dataset[i:i+8]
                    inputs         = {k: torch.tensor(v).to(self.device) for k, v in batch.items() if k != "attention_mask"}
                    
                    outputs        = self.model(**inputs)
                    loss           = outputs.loss
                    
                    total_loss    += loss.item() * inputs["input_ids"].size(0)
                    total_tokens  += inputs["input_ids"].size(0) * inputs["input_ids"].size(1)
            
            perplexity             = torch.exp(torch.tensor(total_loss / total_tokens))
            
            results["perplexity"]  = perplexity.item()
        
        if metric == "coherence" or metric == "all":
            
            print("Calculating coherence...")
            
            self.model.eval()
            total_coherence = 0
            
            with torch.no_grad():
            
                for i in tqdm(range(min(100, len(dataset)))):  
                    sample             = dataset[i]
                    input_ids          = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
                    
                    outputs            = self.model(input_ids)
                    logits             = outputs.logits
                    
                    shift_logits       = logits[:, :-1, :].contiguous()
                    shift_labels       = input_ids[:, 1:].contiguous()
                    
                    log_probs          = torch.log_softmax(shift_logits, dim=-1)
                    token_log_probs    = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    
                    coherence          = token_log_probs.mean().item()
                    total_coherence   += coherence
            
            results["coherence"]       = total_coherence / min(100, len(dataset))   

        if metric == "bleu" or metric == "all":
            
            print("Calculating BLEU score...")

            references       = []
            candidates       = []
            
            for i in tqdm(range(min(100, len(dataset)))):
                sample       = dataset[i]
                

                input_text   = self.tokenizer.decode(sample["input_ids"][:50])  
                
                reference    = self.tokenizer.decode(sample["input_ids"][50:])
                references.append([reference.split()])
                
                output       = self.inference(input_text)[0]
                candidates.append(output.split())

            bleu             = corpus_bleu(references, candidates)
            results["bleu"]  = bleu
        
        print("Evaluation complete:")
        
        for k, v in results.items():
            print(f"{k}: {v}")
        
        return results