from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math
import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model, train_dataset, val_dataset, max_length=512):
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
        self.last_layer_weights = module.weight.clone().detach() if hasattr(module, 'weight') else None
        self.inputs = input[0].clone().detach()                

    def training_step(self, batch, batch_idx):
        pixel_values, labels, _ = batch

        token_sequence_X = labels.tolist()
        print(f"Token sequence during training: {token_sequence_X}")

        input_lengths = (labels != self.processor.tokenizer.pad_token_id).sum(dim=-1)
        for length in input_lengths:
            self.log(f"input_length_freq_{length.item()}", 1, on_step=True, on_epoch=True, reduce_fx="sum")

        try:
            # print("Training pixel values shape:", pixel_values.shape)
            outputs = self.model(pixel_values, labels=labels)
        except RuntimeError as e:
            print(f"Error during model forward pass: {e}")
            raise e

        loss = outputs.loss
        print(f"Training loss: {loss}")
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, labels, answers = batch
        batch_size = pixel_values.shape[0]

        # Log the frequency of input lengths
        input_lengths = (labels != self.processor.tokenizer.pad_token_id).sum(dim=-1)
        for length in input_lengths:
            self.log(f"val_input_length_freq_{length.item()}", 1, on_step=True, on_epoch=True, reduce_fx="sum")

        try:
            outputs = self.model(pixel_values, labels=labels)
        except RuntimeError as e:
            print(f"Error during model forward pass: {e}")
            raise e

        val_loss = outputs.loss
        print(f"Validation loss: {val_loss}")
        self.log("val_loss", val_loss)            

        # feed the prompt to the model
        decoder_input_ids = torch.full((batch_size, 1), self.model.config.decoder_start_token_id, device=self.device)
        
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
        
        token_sequence = outputs.sequences.tolist()
        print(f"Token sequence during validation: {token_sequence }")

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=2).strip()  # remove first task start
            # print("Seq", seq) 
            predictions.append(seq)
        scores = []
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
        model_weights = self.model.state_dict()
        for name, param in model_weights.items():
            self.logger.experiment.log({f"weights/{name}": wandb.Histogram(param.cpu().numpy())})    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.get("train_batch_sizes")[0], shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)