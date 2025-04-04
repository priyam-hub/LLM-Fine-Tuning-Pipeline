# DEPENDENCIES  

import torch
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu

import datasets

import transformers
from transformers import PreTrainedTokenizer

class ModelEvaluator:
    """
    A class for evaluating the performance of a fine-tuned model using different evaluation metrics.

    This class supports metrics such as perplexity, coherence, and BLEU score to assess model performance
    on a given test dataset.
    """
    
    def __init__(self, model : transformers.PreTrainedModel, tokenizer : PreTrainedTokenizer, device : str = "cpu", prepared_dataset : datasets.DatasetDict = None) -> None:
        """
        Initializes the ModelEvaluator with a model, tokenizer, and evaluation dataset.

        Arguments:
            
            `model`           {transformers.PreTrainedModel}         : The fine-tuned model to be evaluated.
            
            `tokenizer`      {transformers.PreTrainedTokenizer}      : The tokenizer corresponding to the model.
            
            `device`                  {str, optional}                : The device to run the evaluation on ("cpu" or "cuda"). Defaults to "cpu".
             
            `prepared_dataset`        {dict, optional}               : The dataset prepared for evaluation, containing train and test splits.
        
        """
        self.model             = model
        self.device            = device
        self.tokenizer         = tokenizer
        self.prepared_dataset  = prepared_dataset if prepared_dataset else {}


    def evaluate(self, test_dataset : datasets.DatasetDict = None, metric : str = "perplexity") -> dict:
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