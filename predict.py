from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import json
import torch
import Levenshtein
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from donut import JSONParseEvaluator

processor = DonutProcessor.from_pretrained("xeko56/html-generator-level-2")
model = VisionEncoderDecoderModel.from_pretrained("xeko56/html-generator-level-2")

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()
model.to(device)

def minify_html(html: str):
    # function to check

    # Replace escaped double quotes with regular double quotes
    html = html.replace('\\"', '"')
    # Remove newline characters
    html = html.replace('\n', '')
    # Optionally, remove extra spaces between tags if they exist
    html = ' '.join(html.split())

    tables = re.findall(r'<table.*?>.*?</table>', html, re.DOTALL)
    cleaned_tables = [re.sub(r'\s*class="[^"]*"', '', table) for table in tables]

    # Join the cleaned tables back into a single HTML string
    html = ''.join(cleaned_tables)

    return html

def levenshtein_ratio(predicted, target):
    """Calculate Levenshtein ratio between two strings as a similarity measure."""
    return 1 - Levenshtein.distance(predicted, target) / max(len(predicted), len(target))

def process_data(json_file, split='validation'):
    with open(json_file, 'r') as file:
        data = json.load(file)
        total_len = len(data)
        train_end = int(0.7 * total_len)
        val_end = int(0.85 * total_len)

        if split == 'train':
            data_pairs = data[:train_end]
        elif split == 'validation':
            data_pairs = data[train_end:val_end]
        elif split == 'test':
            data_pairs = data[val_end:]
        else:
            raise ValueError("Invalid split name")

        accs = []
        output_list = []
        for sample in tqdm(data_pairs, total=len(data_pairs)):
            # prepare encoder inputs
            image_path = sample['image'].replace("\\", "/")
            image = Image.open(image_path).convert('RGB')
            image = processor(image, return_tensors="pt").pixel_values
            image = image.to(device)

            # prepare decoder inputs
            task_prompt = "<html_table>"
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(device)

            # autoregressively generate sequence
            outputs = model.generate(
                image,
                decoder_input_ids=decoder_input_ids,
                max_length=model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )

            # turn into JSON
            seq = processor.batch_decode(outputs.sequences)[0]
            seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=2).strip()  # remove first task start token

            ground_truth = sample["html"]
            ground_truth = minify_html(ground_truth)
            # evaluator = JSONParseEvaluator()
            # score = evaluator.cal_acc(seq, ground_truth)
            print(seq)
            print(ground_truth)
            print(Levenshtein.distance(seq, ground_truth))

            score = levenshtein_ratio(seq, ground_truth)

            print(score)

            #Unit test, 2 or 3 pairs to explain the result

            accs.append(score)
            output_list.append(seq)

        scores = {"accuracies": accs, "mean_accuracy": np.mean(accs)}
        print(scores, f"length : {len(accs)}")
        with open(f'{split}_output_2_l.json', 'w') as file:
            json.dump(scores, file)

json_file = 'data_pairs.json'
process_data(json_file, split='test')