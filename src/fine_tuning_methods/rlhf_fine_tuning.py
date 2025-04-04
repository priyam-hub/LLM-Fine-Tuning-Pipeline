# DEPENDENCIES

import datasets
import datetime
from tqdm import tqdm

import transformers
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from trl import PPOConfig
from trl import PPOTrainer
from trl import create_reference_model
from trl import AutoModelForCausalLMWithValueHead

class RLHFTrainer:
    """
    A class for fine-tuning a language model using Reinforcement Learning from Human Feedback (RLHF)
    with Proximal Policy Optimization (PPO).
    
    """

    def __init__(self, model : AutoModelForCausalLM, tokenizer : AutoTokenizer, prepared_dataset : datasets.DatasetDict) -> None:
        """
        Initializes the RLHFTrainer.

        Arguments:

            `model`              {AutoModelForCausalLM}        : The pretrained language model to be fine-tuned.
            
            `tokenizer`              {AutoTokenizer}           : The tokenizer corresponding to the model.
            
            `prepared_dataset`          {Dataset}              : The dataset containing prompts for training.
        
        """
        self.model             = model
        self.tokenizer         = tokenizer
        self.prepared_dataset  = prepared_dataset

    def apply_rlhf(self, 
                   output_dir       : str = "../../model/rlhf_fine_tuned_model", 
                   reward_model_id  : str = None, 
                   batch_size       : int = 4, 
                   learning_rate    : float = 1e-5, 
                   num_epochs       : int = 1
                   ) -> AutoModelForCausalLM:
        """
        Fine-tunes the model using RLHF with PPO.

        Arguments:

            `output_dir`               {str, optional}             : Directory to save the fine-tuned model. Defaults to "./rlhf_model".
        
            `reward_model_id`          {str, optional}             : Hugging Face model ID for the reward model.
            
            `batch_size`               {int, optional}             : Batch size for PPO training. Defaults to 4.
            
            `learning_rate`           {float, optional}            : Learning rate for PPO. Defaults to 1e-5.
            
            `num_epochs`               {int, optional}             : Number of training epochs. Defaults to 1.

        Raises:
            
            ValueError: If the model or dataset is not loaded.

        Returns:
            
            AutoModelForCausalLM: The fine-tuned model.
        
        """
        if self.model is None or self.prepared_dataset is None:
            raise ValueError("Model and prepared dataset must be loaded first")

        print("Starting RLHF training...")
        
        model                   = AutoModelForCausalLMWithValueHead.from_pretrained(self.model.config._name_or_path 
                                                                                    if hasattr(self.model, "config") 
                                                                                    else self.model.name_or_path,
                                                                                    trust_remote_code = True
                                                                                    )
        ref_model               = create_reference_model(model)

        if reward_model_id:
            reward_model        = AutoModelForCausalLM.from_pretrained(reward_model_id)
            reward_tokenizer    = AutoTokenizer.from_pretrained(reward_model_id)
            
            reward_pipeline     = pipeline("text-generation", 
                                           model      = reward_model, 
                                           tokenizer  = reward_tokenizer
                                           )

        else:
            print("Warning: No reward model provided. Using a simple length-based reward function.")

            def reward_fn(response : list[str]) -> list[float]:
                """
                Computes a simple heuristic-based reward for each response based on its length.

                Arguments:

                    `response`          {list of str}        : A list of generated response texts.

                Returns:
                    
                    list of float: A list of reward scores (between 0 and 1), where each score is calculated as:
                                
                                   - The number of words in the response divided by 50.
                                   
                                   - Capped at a maximum of 1.0 for responses with 50 or more words.

                Functionality:

                    - Splits each response into words and counts them.
                    - Normalizes the count by dividing by 50.
                    - Ensures the reward does not exceed 1.0.

                Example:

                    >>> reward_fn(["This is a sample response with ten words only."])
                    [0.2]  # (10 words / 50)
                
                """

                return [min(len(r.split()), 50) / 50.0 for r in response]

        ppo_config                = PPOConfig(batch_size      = batch_size,
                                              learning_rate   = learning_rate,
                                              ppo_epochs      = num_epochs,
                                              model_name      = model.config._name_or_path 
                                              if hasattr(model, "config") 
                                              else model.name_or_path,
                                              )
        
        ppo_trainer               = PPOTrainer(config      = ppo_config, 
                                               model       = model, 
                                               ref_model   = ref_model, 
                                               tokenizer   = self.tokenizer
                                               )

        gen_kwargs                = {"max_new_tokens": 100, "temperature": 0.7, "top_p": 0.9}
        prompts                   = [example["input_text"] for example in self.prepared_dataset["train"]]

        total_start_time          = datetime.datetime.now()
        
        print(f"Fine-tuning started at: {total_start_time.strftime('%H:%M:%S')}")
        print("Training model...")

        for epoch in range(1, num_epochs + 1):
            
            epoch_start_time      = datetime.datetime.now()
            
            print(f"\nEpoch {epoch}/{num_epochs} started at {epoch_start_time.strftime('%H:%M:%S')}")

            with tqdm(total       = len(prompts) // batch_size, 
                      desc        = f"Epoch {epoch}/{num_epochs}", 
                      bar_format  = "{l_bar}{bar} | {n_fmt}/{total_fmt} [Time Left: {remaining}]", 
                      ncols       = 80
                      ) as pbar:
                
                for i, prompt in enumerate(prompts[:100]):  # Limit for demonstration
                    response_tensors   = ppo_trainer.generate(prompt, **gen_kwargs)
                    response_texts     = [self.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                    rewards            = [reward_pipeline(r)[0]["score"] for r in response_texts] if reward_model_id else reward_fn(response_texts)
                    stats              = ppo_trainer.step(response_tensors, rewards)

                    if i % 10 == 0:
                        print(f"Batch {i} stats: {stats}")
                    pbar.update(1)

            epoch_end_time             = datetime.datetime.now()
            time_taken                 = epoch_end_time - epoch_start_time
            print(f"Epoch {epoch} finished at {epoch_end_time.strftime('%H:%M:%S')} | Time Taken: {str(time_taken)}")

        total_end_time                 = datetime.datetime.now()
        total_time_taken               = total_end_time - total_start_time
        
        print("\nFine-tuning Completed!")
        print(f"Total Training Time: {str(total_time_taken)}")
        print(f"Started at: {total_start_time.strftime('%H:%M:%S')}")
        print(f"Finished at: {total_end_time.strftime('%H:%M:%S')}")

        ppo_trainer.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to: {output_dir}")

        return self.model
