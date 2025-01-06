import json
import random
import torch
import re
from typing import Any, List, Tuple
from utils import minify_html
from transformers import PreTrainedTokenizer, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class HtmlTablesDataset(Dataset):
    """
    A PyTorch Dataset class for handling HTML table data. This class is designed to preprocess and structure data from 
    a given JSON file containing HTML tables, preparing it for use in models that involve both vision (image) and 
    text (HTML) modalities. It integrates with a tokenizer and a VisionEncoderDecoderModel to enable processing of 
    the data for tasks like table structure recognition or information extraction.

    Args:
        json_file (str): Path to the JSON file containing HTML table data.
        processor (Any): A processor object responsible for any additional preprocessing of the input data.
        tokenizer (PreTrainedTokenizer): Tokenizer used for encoding the input HTML data.
        model (VisionEncoderDecoderModel): Pre-trained model that encodes images and decodes the table structure.
        max_length (int): Maximum sequence length for the tokenized input.
        split (str, optional): Dataset split to load, such as 'train', 'test', or 'validation'. Default is 'train'.
        ignore_id (int, optional): Token ID used to ignore certain tokens during model training (e.g., padding tokens). Default is -100.
        task_start_token (str, optional): The start token used for model task-specific prompting. Default is "<s>".
        prompt_end_token (str, optional): Token indicating the end of the prompt, if applicable. Default is None.
        sort_json_key (bool, optional): Whether to sort the keys in the JSON file before processing. Default is True.
        added_tokens (List[str], optional): Additional tokens that should be added to the tokenizer. Default is an empty list.
    """
    def __init__(
        self,
        json_file: str,
        processor: Any,
        tokenizer: PreTrainedTokenizer,
        model: VisionEncoderDecoderModel,
        max_length: int,
        split: str = "train",
        ignore_id: int = -100,
        task_start_token: str = "<s>",
        prompt_end_token: str = None,
        sort_json_key: bool = True,
        added_tokens: List[str] = [],
    ):
        super().__init__()
        self.processor = processor
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length
        self.split = split
        self.ignore_id = ignore_id
        self.task_start_token = task_start_token
        self.prompt_end_token = prompt_end_token if prompt_end_token else task_start_token
        self.sort_json_key = sort_json_key
        self.added_tokens = added_tokens

        with open(json_file, 'r') as file:
            # Split data
            data = json.load(file)
            total_len = len(data)
            train_end = int(0.7 * total_len)
            val_end = int(0.85 * total_len)
            
            if self.split == 'train':
                self.data_pairs = data[:train_end]
            elif self.split == 'validation':
                self.data_pairs = data[train_end:val_end]
            elif self.split == 'test':
                self.data_pairs = data[val_end:]
            else:
                raise ValueError("Invalid split name")             
        self.dataset_length = len(self.data_pairs)       

        # Process the ground truth HTML data
        self.gt_token_sequences = []
        for sample in self.data_pairs:
            gt_jsons = sample["html"]
            # Add the task start token and prompt end token to the HTML content
            self.gt_token_sequences.append([self.minify_html(gt_jsons) + self.tokenizer.eos_token])

        # Add the special tokens to the tokenizer
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
    
    def add_tokens(self, list_of_tokens: List[str]):
        """
        Add special tokens to tokenizer and resize the token embeddings of the decoder
        """
        newly_added_num = self.tokenizer.add_tokens(list_of_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.tokenizer))
            self.added_tokens.extend(list_of_tokens)
    
    def __len__(self) -> int:
        return len(self.data_pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.data_pairs[idx]
        image_path = item['image'].replace("\\", "/")
        image = Image.open(image_path).convert('RGB')

        pixel_values = self.processor(image, random_padding=self.split == "train", return_tensors="pt").pixel_values
        if pixel_values.ndim != 4:
            raise ValueError(f"Unexpected pixel_values shape: {pixel_values.shape}")

        pixel_values = pixel_values.squeeze()    

        # Get the target sequence, which is a HTML string and encode it
        target_sequence = self.gt_token_sequences[idx]

        encoded_html = self.tokenizer(
            target_sequence,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )["input_ids"].squeeze(0)

        # Replace padding tokens with the ignore_id
        labels = encoded_html.clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_id
    
        return pixel_values, labels, target_sequence