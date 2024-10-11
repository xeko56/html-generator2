import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
import re
import torch
import hashlib
import os
from tqdm.auto import tqdm
import numpy as np
from donut import JSONParseEvaluator
from model import DonutModelPLModule
from dataset import HtmlTablesDataset

st.set_page_config(layout="wide")

def get_model_hash(model):
    """Compute the hash of the model's state dict."""
    state_dict = model.state_dict()
    model_bytes = b''.join(param.cpu().numpy().tobytes() for param in state_dict.values())
    return hashlib.md5(model_bytes).hexdigest()

def generate_html_from_image(image, checkpoint_path=""):
    # ./save_pretrained
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "xeko56/html-generator"

    processor = DonutProcessor.from_pretrained("xeko56/html-generator-level-1")
    model = VisionEncoderDecoderModel.from_pretrained("xeko56/html-generator-level-1")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    state_dict_hash = get_model_hash(model)
    print(f"Loaded model state dict hash: {state_dict_hash}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    image = image.convert('RGB')
    image = processor(image, return_tensors="pt").pixel_values
    image = image.to(device)

    task_prompt = "<html_table>"
    # decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    # decoder_input_ids = decoder_input_ids.to(device)
    decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)
    print(decoder_input_ids)

    outputs = model.generate(image,
                                decoder_input_ids=decoder_input_ids,
                                max_length=768,
                                early_stopping=True,
                                pad_token_id=processor.tokenizer.pad_token_id,
                                eos_token_id=processor.tokenizer.eos_token_id,
                                use_cache=True,
                                num_beams=1,
                                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                return_dict_in_generate=True,)    

    # outputs = model.generate(
    #     image,
    #     decoder_input_ids=decoder_input_ids,
    #     max_length=model.decoder.config.max_position_embeddings,
    #     early_stopping=True,
    #     pad_token_id=processor.tokenizer.pad_token_id,
    #     eos_token_id=processor.tokenizer.eos_token_id,
    #     use_cache=True,
    #     num_beams=1,
    #     bad_words_ids=[[processor.tokenizer.unk_token_id]],
    #     do_sample=False,  # Enable sampling for more diversity
    #     top_k=None,  # Use top-k sampling to only sample from the top 50 tokens
    #     top_p=None,  # Or use nucleus sampling (sampling from the top 90% probability mass)
    #     temperature=None,  # Lower temperature for more focused sampling
    #     return_dict_in_generate=True,      
    # )

    token_sequence_Y = outputs.sequences.tolist()
    print(f"Token sequence during validation: {token_sequence_Y }")
    decoded_validation_sequences = processor.tokenizer.batch_decode(token_sequence_Y, skip_special_tokens=True)
    print(f"Decoded validation sequences: {decoded_validation_sequences}")
    seq = processor.batch_decode(outputs.sequences)[0]
    print(seq) 
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=2).strip()
            

    return seq

def main():
  st.title("HTML Generator from Image")

  col1, col2 = st.columns(2)

  with col1:
      uploaded_image = st.file_uploader("Upload an Image of an HTML Table", type=["jpg", "png", "jpeg"])

      if uploaded_image is not None:
          image = Image.open(uploaded_image)
          st.image(image, caption="Uploaded Image", use_column_width=True)
  
  # Displaying output in the second column
  with col2:
      if uploaded_image is not None:
          html_code = generate_html_from_image(image)
          
          # Option to choose between HTML Code or Rendered HTML
          display_option = st.radio(
              "Output",
              ("HTML Code", "Rendered HTML")
          )

          if display_option == "HTML Code":
              st.subheader("Generated HTML Code")
              st.code(html_code, language='html')
          else:
              st.subheader("Rendered HTML Table")
              st.markdown(html_code, unsafe_allow_html=True)

# streamlit run html_table_generator.py

if __name__ == "__main__":
    main()
