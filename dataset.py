import json
import random
import torch
import re
from typing import Any, List, Tuple
from transformers import PreTrainedTokenizer, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class HtmlTablesDataset(Dataset):
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

        # Initialize transformations for images

        # html_tokens = ['<table>', '<table style="border-collapse: collapse;">', '<th>'
        #                , '<th style="border: 1px solid black;">', '<tr>', '<td>', '</td>'
        #                , '<td style="border: 1px solid black;">', '</tr>', '</th>', '</table>', '<s_html>']
        # self.add_tokens(html_tokens)

        self.gt_token_sequences = []
        for sample in self.data_pairs:
            gt_jsons = sample["html"]
            self.gt_token_sequences.append([self.minify_html(gt_jsons) + self.tokenizer.eos_token])
        self.add_tokens([self.task_start_token, self.prompt_end_token])
        self.prompt_end_token_id = self.tokenizer.convert_tokens_to_ids(self.prompt_end_token)
        # print("tokenizer", self.tokenizer)

    def minify_html(self, html: str):
        # function to check

        # Replace escaped double quotes with regular double quotes
        html = html.replace('\\"', '"')
        # Remove newline characters
        html = html.replace('\n', '')
        # Optionally, remove extra spaces between tags if they exist
        html = ' '.join(html.split())

        tables = re.findall(r'<table.*?>.*?</table>', html, re.DOTALL)
        html = ''.join(tables)
        
        return html

    def json2token(self, obj: Any, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        self.add_tokens([fr"", fr""])
                    output += (
                        fr""
                        + self.json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr""
                    )
                return output
        elif type(obj) == list:
            return r"".join(
                [self.json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            obj = str(obj)
            if f"<{obj}/>" in self.added_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj
    
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

        target_sequence = random.choice(self.gt_token_sequences[idx])
        # print(f"index: {idx}", f"self.gt_token_sequences[idx]: {self.gt_token_sequences[idx]}")

        encoded_html = self.tokenizer(
            target_sequence,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )["input_ids"].squeeze(0)

        # print(html_content)
        # print("Encoded HTML input IDs:", encoded_html)

        labels = encoded_html.clone()
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_id
    
        return pixel_values, labels, target_sequence
    
    def reassemble_html_tokens(self, tokens):
        # Implement reassembly logic that was discussed previously
        new_tokens = []
        buffer = ""
        for token in tokens:
            if token.startswith("▁") and buffer:
                new_tokens.append(buffer)
                buffer = token[1:]  # Remove the '▁' for a new token
            else:
                buffer += token.replace("▁", "")  # Remove '▁' and append to the current buffer
        if buffer:
            new_tokens.append(buffer)  # Append the last buffer if any
        return new_tokens    