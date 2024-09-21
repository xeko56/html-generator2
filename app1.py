import streamlit as st
from PIL import Image
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
import re
import torch
from tqdm.auto import tqdm
import numpy as np
from donut import JSONParseEvaluator
from model import DonutModelPLModule
from dataset import HtmlTablesDataset

st.set_page_config(layout="wide")

# Simulating a function that converts image to HTML (you will replace this with your actual model)
def generate_html_from_image(image, checkpoint_path="./model_checkpoints/last-v3.ckpt"):
    # processor = DonutProcessor.from_pretrained("xeko56/html-demo")

    image_size = [512, 512]
    max_length = 768

    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    original_model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)
    tokenizer = processor.tokenizer
    added_tokens = []

    config = {
        "max_epochs":30,
        "val_check_interval":0.2, # how many times we want to validate during an epoch
        "check_val_every_n_epoch":1,
        "gradient_clip_val":1.0,
        "num_training_samples_per_epoch": 800,
        "lr":1e-6,
        "train_batch_sizes": [1],
        "val_batch_sizes": [1],
        # "seed":2022,
        "num_nodes": 1,
        "warmup_steps": 300, # 800/8*30/10, 10%
        "result_path": "./result",
        "verbose": True,
    }
    train_dataset = HtmlTablesDataset(
        processor=processor,
        json_file='./data_pairs.json',
        tokenizer=tokenizer,
        model=original_model, 
        max_length=max_length,                         
        split="train", 
        task_start_token="<s_html>", 
        prompt_end_token="<s_html>",
        ignore_id=-100,
        added_tokens=added_tokens,
    )

    val_dataset = HtmlTablesDataset(
        processor=processor,
        json_file='./data_pairs.json',
        tokenizer=tokenizer,
        model=original_model,
        max_length=max_length,                         
        split="validation", 
        task_start_token="<s_html>", 
        prompt_end_token="<s_html>",
        ignore_id=-100,
        added_tokens=added_tokens,
    )        

    # model = VisionEncoderDecoderModel.from_pretrained("xeko56/html-demo")
    model = DonutModelPLModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        processor=processor,
        model=original_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    image = image.convert('RGB')
    image = processor(image, return_tensors="pt").pixel_values
    image = image.to(device)

    task_prompt = "<s_html>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    outputs = model.model.generate(
        image,
        decoder_input_ids=decoder_input_ids,
        max_length=model.model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        do_sample=False,
        top_k=None,
        top_p=None,
        temperature=None,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=3).strip()         

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
