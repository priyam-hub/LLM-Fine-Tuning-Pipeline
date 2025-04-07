# DEPENDENCIES

from math import e
import re
import datasets
from re import A

from sympy import N
import transformers
from transformers import PreTrainedTokenizer
from transformers import AutoModelForCausalLM

from ..utils.logger import LoggerSetup

dataPreparation_logger = LoggerSetup(logger_name = "prepare_dataset.py", log_filename_prefix = "prepare_dataset").get_logger()


class DatasetPreparer:
    """
    A class for preparing datasets for fine-tuning language models.

    This class supports two modes of dataset preparation:

    1. **Instruction Fine-Tuning**: If `instruction_column` and `response_column` are provided,  
       the dataset is formatted with input-output pairs based on the `prompt_template`.
    
    2. **Regular Language Model Fine-Tuning**: If `text_column` is provided, the dataset is tokenized  
       as a standard language modeling dataset.
    """

    def __init__(self, dataset: datasets.DatasetDict, tokenizer : transformers.PreTrainedTokenizer, model : AutoModelForCausalLM) -> None:
        """
        Initializes the DatasetPreparer class.

        Arguments:

            `dataset`     {datasets.DatasetDict}  : The raw dataset to be prepared for fine-tuning.
            
            `tokenizer`   {transformers.PreTrainedTokenizer} : The tokenizer used for tokenization.
            
            `model`       {transformers.PreTrainedModel}     : The model to determine tokenization behavior.
        
        """

        try:
            self.model             = model
            self.dataset           = dataset
            self.tokenizer         = tokenizer
            self.prepared_dataset  = None

            dataPreparation_logger.info("DatasetPreparer initialized successfully") 

        except Exception as e:
            dataPreparation_logger.error(dataPreparation_logger, f"Error initializing DatasetPreparer: {repr(e)}")
            
            raise e

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
            dataPreparation_logger.error("Tokenizer and dataset must be loaded before preparing the dataset")
            
            raise ValueError("Model tokenizer and dataset must be loaded first")
        
        dataPreparation_logger.info("Preparing dataset for fine-tuning...")
        
        # FOR INSTRUCTION FINE-TUNING
        
        if instruction_column and response_column:

            dataPreparation_logger.info("Preparing dataset for instruction fine-tuning...")
        
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

                try:

                    label_map                  = {0: "negative", 1: "positive"}
                    sentiment                  = label_map.get(example.get("label", -1), "unknown")
                    
                    if prompt_template:
                        prompt                 = prompt_template.format(instruction = example["text"])
                        example["input_text"]  = prompt
                        example["output_text"] = sentiment
            
                    else:
                        example["input_text"]  = f"Given the following movie review, determine if the sentiment is positive or negative:\n\nReview: {example['text']}\nSentiment:"
                        example["output_text"] = sentiment

                    dataPreparation_logger.debug("Example Formatted")

                except KeyError as e:
                    dataPreparation_logger.error(f"KeyError in format_instruction: {repr(e)}")
                    
                    raise e
     
                return example
            
            formatted_dataset              = self.dataset.map(format_instruction)
            
            def tokenize_function(examples : dict) -> dict:
                """
                Tokenizes input and output text for fine-tuning language models.

                Arguments:

                    examples                 {dict}           : A batch of dataset examples containing:
                        
                        - `"input_text"`      {str}           : The input text (e.g., an instruction prompt).
                        
                        - `"output_text"` {str, optional}     : The expected response text (for encoder-decoder models).

                Returns:

                    dict: A dictionary containing tokenized inputs with the following keys:
                        
                        - `"input_ids"`        : Tokenized input sequences.
                        
                        - `"attention_mask"`   : Attention masks for padding.
                        
                        - `"labels"`           : Tokenized output sequences (for supervised fine-tuning).

                Behavior:

                    - **For decoder-only models (`AutoModelForCausalLM`)**: 
                    - The `labels` are set to the same value as `input_ids` (self-supervised learning).
                    - **For encoder-decoder models**:
                    - The `input_text` is tokenized separately for input.
                    - The `output_text` is tokenized to create the `labels`.

                """

                try:
        
                    model_inputs               = self.tokenizer(examples["input_text"], 
                                                                truncation      = True, 
                                                                max_length      = max_length, 
                                                                padding         = "max_length",
                                                                return_tensors  = None,
                                                                )
                    
                    # FOR DECODER-ONLY MODELS
                    if isinstance(self.model, AutoModelForCausalLM):
                        model_inputs["labels"] = model_inputs["input_ids"].copy()

                    # FOR ENCODER-ONLY MODELS
                    else:
                        
                        labels                 = self.tokenizer(examples["output_text"],
                                                                truncation      = True,
                                                                max_length      = max_length,
                                                                padding         = "max_length",
                                                                return_tensors  = None,
                                                                )
                        
                        model_inputs["labels"] = labels["input_ids"]

                        
                    return model_inputs
                
                except Exception as e:
                    dataPreparation_logger.error(f"Error in tokenize_function: {repr(e)}")
                    
                    raise e
        
        # FOR REGULAR LANGUAGE MODEL FINE-TUNING
        else:

            dataPreparation_logger.info("Preparing dataset for regular language model fine-tuning...")

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
                                      truncation      = True,
                                      max_length      = max_length,
                                      padding         = "max_length",
                                      return_tensors  = None,
                                      )
        

        try:

            dataPreparation_logger.info("Tokenizing dataset...")
        
            tokenized_dataset             = formatted_dataset.map(tokenize_function,
                                                                batched         = True,
                                                                remove_columns  = ["text", "label", "input_text", "output_text"],
                                                                )
            
            self.prepared_dataset         = tokenized_dataset
            
            dataPreparation_logger.info("Dataset preparation completed")
            
            return self.prepared_dataset
        
        except Exception as e:
            dataPreparation_logger.error(f"Error in prepare_dataset: {repr(e)}")
            
            raise e