# DEPENDENCIES

from re import split

from src.utils.model_loader import ModelLoader
from src.utils.dataset_loader import DatasetLoader

from src.data_preparation.prepare_dataset import DatasetPreparer

from src.fine_tuning_methods.rlhf_fine_tuning import RLHFTrainer
from src.fine_tuning_methods.lora_fine_tuning import LoRAFineTuning
from src.fine_tuning_methods.supervised_fine_tuning import SupervisedFineTuning
from src.fine_tuning_methods.instruction_fine_tuning import InstructionFineTuning


def main():

    model_loader      = ModelLoader()  
    dataset_loader    = DatasetLoader()
    
    model_id          = "google-bert/bert-base-uncased"
    model, tokenizer  = model_loader.download_model(model_id)
    
    dataset           = dataset_loader.load_dataset("imdb")

    dataset_preparer  = DatasetPreparer(model = model, tokenizer = tokenizer, dataset = dataset)

    prompt_template   = "Given the following movie review, determine if the sentiment is positive or negative:\n\nReview: {instruction}\nSentiment:"


    tokenized_dataset = dataset_preparer.prepare_dataset(instruction_column = "text", 
                                                         response_column    = "label",    
                                                         prompt_template    = prompt_template,
                                                         )
    
    # # LoRA FINE-TUNING
    
    # lora_adapter      = LoRAFineTuning(model = model)
    
    # quantized_model   = lora_adapter.apply_lora(rank            = 8, 
    #                                             lora_alpha      = 16, 
    #                                             lora_dropout    = 0.1,
    #                                             target_modules  = ["query", "key", "value"]
    #                                             )


    # INSTRUCTION FINE-TUNING

    fine_tuner        = InstructionFineTuning(model = model, tokenizer = tokenizer, dataset = tokenized_dataset)

    fine_tuned_model  = fine_tuner.apply_instruction_fine_tuning(output_dir      = "./model/instruction_fine_tuned_model",
                                                                 batch_size      = 8,
                                                                 learning_rate   = 5e-5,
                                                                 num_epochs      = 3
                                                                 )

    # fine_tuner        = SupervisedFineTuning(model = model, tokenizer = tokenizer, dataset = tokenized_dataset)

    # # SUPERVISED FINE-TUNING

    # fine_tuned_model  = fine_tuner.apply_supervised_fine_tuning(output_dir      = "./model/supervised_fine_tuned_model",
    #                                                              batch_size      = 8,
    #                                                              learning_rate   = 2e-5,
    #                                                              num_epochs      = 3
    #                                                              )
    
    # # RLHF FINE-TUNING

    # fine_tuner        = RLHFTrainer(model=model, tokenizer=tokenizer, prepared_dataset=tokenized_dataset)

    # fine_tuned_model  = fine_tuner.apply_rlhf(output_dir       = "./model/rlhf_fine_tuned_model",
    #                                           reward_model_id  = None,
    #                                           batch_size       = 4,
    #                                           learning_rate    = 1e-5,
    #                                           num_epochs       = 1)
    


if __name__ == "__main__":
    main()