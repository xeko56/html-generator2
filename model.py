from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

class DonutModelPLModule(pl.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating the Donut model, a VisionEncoderDecoder model used for tasks 
    like document understanding or table extraction.

    Args:
        config (dict): Configuration settings for the model and training process.
        processor (Any): Preprocessing object responsible for preparing the input data, including tokenization and image processing.
        model (VisionEncoderDecoderModel): The Donut model, a multi-modal architecture that processes both visual and textual data.
        train_dataset (Dataset): Dataset used for training the model.
        val_dataset (Dataset): Dataset used for validating the model's performance during training.
        max_length (int, optional): Maximum sequence length for tokenized inputs. Default is 1000.

    Attributes:
        last_layer (Module): The last layer of the Donut model, which is identified and used for logging or monitoring purposes.
        last_layer_weights (torch.Tensor or None): Stores the weights of the last layer, captured during the forward pass.
        inputs (torch.Tensor or None): Stores the inputs to the last layer, captured during the forward pass.
        
    Methods:
        save_last_layer_weights_and_inputs: A hook function registered to the last layer, which captures its weights and inputs during the forward pass. 
        This is used for monitoring or analysis during training.
    """
    def __init__(self, config, processor, model, train_dataset, val_dataset, max_length=1000):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.max_length = max_length

        # Identify the last layer of the model for logging
        self.last_layer = list(self.model.modules())[-1]
        
        # Storage for weights and inputs
        self.last_layer_weights = None
        self.inputs = None
        
        # Register hook to capture the weights and inputs of the last layer
        self.last_layer.register_forward_hook(self.save_last_layer_weights_and_inputs)

    def save_last_layer_weights_and_inputs(self, module, input, output):
        """
        Save the weights and inputs of the last layer of the model.
        """
        self.last_layer_weights = module.weight.clone().detach() if hasattr(module, 'weight') else None
        self.inputs = input[0].clone().detach()                

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch
        try:
            # Forward pass
            outputs = self.model(pixel_values, labels=labels)
        except RuntimeError as e:
            print(f"Error during model forward pass: {e}")
            raise e
        
        # Compute the loss
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]

        try:
            outputs = self.model(pixel_values, labels=labels)
        except RuntimeError as e:
            print(f"Error during model forward pass: {e}")
            raise e

        # Compute the loss
        val_loss = outputs.loss
        self.log("val_loss", val_loss)            

        # Feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)
        
        # Generate the HTML code from the image
        outputs = self.model.generate(pixel_values,
                                   decoder_input_ids=decoder_input_ids,
                                   max_length=self.max_length,
                                   early_stopping=True,
                                   pad_token_id=self.processor.tokenizer.pad_token_id,
                                   eos_token_id=self.processor.tokenizer.eos_token_id,
                                   use_cache=True,
                                   num_beams=1,
                                   bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                   return_dict_in_generate=True,)
        

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            # Remove special tokens and extra spaces
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            # Remove the first task start token
            seq = re.sub(r"<.*?>", "", seq, count=2).strip()
            predictions.append(seq)
        scores = []

        # Compute the edit distance between the predictions and the answers
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            scores.append(edit_distance(pred, answer) / max(len(pred), len(answer)))

            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))
        
        return {"val_loss": val_loss, "val_edit_distance": np.mean(scores)}
    
    def on_epoch_end(self):
        # Log the weights of the last layer
        model_weights = self.model.state_dict()
        for name, param in model_weights.items():
            self.logger.experiment.log({f"weights/{name}": wandb.Histogram(param.cpu().numpy())})    

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.get("train_batch_sizes")[0], shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)