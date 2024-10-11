from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
import json
import torch
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from donut import JSONParseEvaluator

processor = DonutProcessor.from_pretrained("xeko56/html-generator-level-3")
model = VisionEncoderDecoderModel.from_pretrained("xeko56/html-generator-level-3")

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

print(processor.tokenizer)

print("Original number of tokens:", processor.tokenizer.vocab_size)
print("Number of tokens after adding special tokens:", len(processor.tokenizer))

device = "cuda" if torch.cuda.is_available() else "cpu"

model.eval()
model.to(device)

image = Image.open('./images/table_4018.png').convert('RGB')
image = processor(image, return_tensors="pt").pixel_values
image = image.to(device)
# prepare decoder inputs
task_prompt = "<html_table>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
print(decoder_input_ids)
decoder_input_ids = decoder_input_ids.to(device)


print(decoder_input_ids)

sampling_methods = [
    {'method': 'beam_search', 'num_beams': 1, 'do_sample': False, 'top_k': None, 'top_p': None, 'temperature': None},
    {'method': 'top_k_sampling', 'num_beams': 1, 'do_sample': True, 'top_k': 50, 'top_p': None, 'temperature': None},
    {'method': 'top_p_sampling', 'num_beams': 1, 'do_sample': True, 'top_k': None, 'top_p': 0.9, 'temperature': None},
    {'method': 'combined_sampling', 'num_beams': 1, 'do_sample': True, 'top_k': 50, 'top_p': 0.9, 'temperature': None},
    {'method': 'temperature_sampling', 'num_beams': 1, 'do_sample': True, 'top_k': None, 'top_p': None, 'temperature': 0.7},
]

# Define the log file path
log_file_path = './generation_log.txt'

# Open the log file in write mode
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    for method in sampling_methods:
        log_file.write(f"Testing {method['method']}\n")

        print(processor.tokenizer)

        # Generate the sequence using the specified method
        outputs = model.generate(
            image,
            decoder_input_ids=decoder_input_ids,
            max_length=768,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=method['num_beams'],
            do_sample=method['do_sample'],
            top_k=method['top_k'],
            top_p=method['top_p'],
            temperature=method['temperature'],
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode and clean up the generated sequence
        seq = processor.batch_decode(outputs.sequences)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        seq = re.sub(r"<.*?>", "", seq, count=2).strip()  # remove first task start token
        
        # Write the output to the log file
        log_file.write(f"Output using {method['method']}:\n")
        log_file.write(seq + "\n")
        log_file.write("\n" + "="*50 + "\n\n")

print(f"Generation log saved to {log_file_path}")
