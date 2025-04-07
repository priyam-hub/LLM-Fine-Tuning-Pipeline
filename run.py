# DEPENDENCIES

import torch
import datasets
from re import split

from config.config import Config
from src.utils.logger import LoggerSetup
from src.utils.model_loader import ModelLoader
from src.utils.dataset_loader import DatasetLoader
from src.llm_inference.llm_inference import InferenceEngine
from src.llm_evaluation.llm_evaluation import ModelEvaluator
from src.data_preparation.prepare_dataset import DatasetPreparer
from src.fine_tuning_methods.rlhf_fine_tuning import RLHFTrainer
from src.fine_tuning_methods.lora_fine_tuning import LoRAFineTuning
from src.fine_tuning_methods.supervised_fine_tuning import SupervisedFineTuning
from src.fine_tuning_methods.instruction_fine_tuning import InstructionFineTuning



def main():

    # INITIALIZING THE CLASS

    model_loader                                    = ModelLoader()  
    dataset_loader                                  = DatasetLoader()
    
    # LOADING THE MODEL

    model_id                                        = Config.HUGGING_FACE_MODEL_ID
    model, tokenizer                                = model_loader.download_model(model_id)

    # LOADING THE DATASET
    
    dataset                                         = dataset_loader.load_dataset(Config.HUGGING_FACE_DATASET)

    # PREPARATION OF THE DATASET INTO TRAINING AND TESTING SETS
 
    dataset_preparer                                = DatasetPreparer(model      = model, 
                                                                      tokenizer  = tokenizer, 
                                                                      dataset    = dataset
                                                                      )
 
    prompt_template                                 = "Given the following movie review, determine if the sentiment is positive or negative:\n\nReview: {instruction}\nSentiment:"


    tokenized_dataset                               = dataset_preparer.prepare_dataset(instruction_column = "text", 
                                                                                       response_column    = "label",    
                                                                                       prompt_template    = prompt_template,
                                                                                       )

    # INSTRUCTION FINE-TUNING

    instruction_fine_tuner                          = InstructionFineTuning(model      = model, 
                                                                            tokenizer  = tokenizer, 
                                                                            dataset    = tokenized_dataset
                                                                            )

    instruction_FT_model, instruction_FT_tokenizer  = instruction_fine_tuner.apply_instruction_fine_tuning(output_dir      = Config.INSTRUCTION_FINE_TUNED_MODEL_PATH,
                                                                                                           batch_size      = 8,
                                                                                                           learning_rate   = 5e-5,
                                                                                                           num_epochs      = 3
                                                                                                           )

    SUPERVISED FINE-TUNING

    supervised_fine_tuner                           = SupervisedFineTuning(model      = model, 
                                                                           tokenizer  = tokenizer, 
                                                                           dataset    = tokenized_dataset
                                                                           )

    supervised_FT_model, supervised_FT_tokenizer    = supervised_fine_tuner.apply_supervised_fine_tuning(output_dir      = Config.SUPERVISED_FINE_TUNED_MODEL_PATH,
                                                                                                         batch_size      = 8,
                                                                                                         learning_rate   = 2e-5,
                                                                                                         num_epochs      = 3
                                                                                                         )
    
    RLHF FINE-TUNING

    rlhf_fine_tuner                                 = RLHFTrainer(model             = model, 
                                                                  tokenizer         = tokenizer, 
                                                                  prepared_dataset  = tokenized_dataset
                                                                  )

    rlhf_FT_model, rlhf_FT_tokenizer                = rlhf_fine_tuner.apply_rlhf(output_dir       = Config.RLHF_FINE_TUNED_MODEL_PATH,
                                                                                 reward_model_id  = None,
                                                                                 batch_size       = 4,
                                                                                 learning_rate    = 1e-5,
                                                                                 num_epochs       = 1
                                                                                 )
    
    LLM - INFERENCE

    inference_engine                                = InferenceEngine(model      = model, 
                                                                      tokenizer  = tokenizer, 
                                                                      device     = "cuda" if torch.cuda.is_available() else "cpu"
                                                                      )

    sample_prompt                                   = prompt_template.format(instruction = dataset["test"][0]["text"])

    generated_outputs                               = inference_engine.inference(prompt                = sample_prompt, 
                                                                                 max_length            = 512, 
                                                                                 temperature           = 0.7, 
                                                                                 num_return_sequences  = 3
                                                                                 )

    for idx, output in enumerate(generated_outputs):
        print(f"\nGenerated Output {idx + 1}:\n{output}")

    # LLM - EVALUATION

    evaluator                                       = ModelEvaluator(model             = model,
                                                                     tokenizer         = tokenizer,
                                                                     device            = "cuda" if torch.cuda.is_available() else "cpu",
                                                                     prepared_dataset  = tokenized_dataset
                                                                     )


    results                                         = evaluator.evaluate(metric = "all")


if __name__ == "__main__":
    main()
