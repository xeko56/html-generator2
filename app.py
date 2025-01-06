import streamlit as st
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from utils import levenshtein_ratio, minify_html
import re
import torch
import os
import Levenshtein
import numpy as np

# Set the page layout to wide
st.set_page_config(layout="wide")

def generate_html_from_image(image):
    """Generate HTML code from an image of an HTML table."""

    # Load the model and processor from the Hugging Face Hub
    processor = DonutProcessor.from_pretrained("xeko56/html-generator-level-4")
    model = VisionEncoderDecoderModel.from_pretrained("xeko56/html-generator-level-4")

    # Remove cache and reset memory to prevent issue by inference
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)

    image = image.convert('RGB')
    image = processor(image, return_tensors="pt").pixel_values
    image = image.to(device)

    # Define the task prompt
    task_prompt = "<html_table>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    # Generate HTML code from the image
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
    # Decode the model output
    seq = processor.batch_decode(outputs.sequences)[0]
    # Remove special tokens and extra spaces
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    # Remove the first task start token
    seq = re.sub(r"<.*?>", "", seq, count=2).strip()

    return seq

def quantitative_evaluation(ground_truth, output):
    """Compute quantitative evaluation metrics for the model."""
    # Compute Levenshtein ratio
    acc = levenshtein_ratio(ground_truth, output)
    levenshtein = Levenshtein.distance(ground_truth, output)
    return {
        "acc": acc,
        "levenshtein": levenshtein
    }

def main():
    st.title("HTML Generator from Image")

    col1, col2 = st.columns(2)
    # Displaying input in the first column
    with col1:
        uploaded_image = st.file_uploader("Upload an Image of an HTML Table", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            # Crop the image for the cases it has not been cropped to bounding box before 
            image = Image.open(uploaded_image)
            bbox = image.getbbox()
            cropped_image = image.crop(bbox)
            # Save or display the cropped image
            cropped_image.save("test.png")
            st.image(cropped_image, caption="Uploaded Image", use_column_width=True)

    # Displaying output in the second column
    with col2:
        if uploaded_image is not None:
            html_code = generate_html_from_image(cropped_image)

            # Please comment out this block for testing on data not coming from the dataset
            # Begin of block

            file_name = os.path.splitext(uploaded_image.name)[0]
            # Load the HTML content from the dataset
            html_file_name = f"dataset_stage4/tables/{file_name}.html"
            with open(html_file_name, 'r') as html_file:
                input_content = html_file.read()
                input_content = minify_html(input_content)

            #End of block
            
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

            # Please comment out this block for testing on data not coming from the dataset
            # Begin of block
            result = quantitative_evaluation(input_content, html_code)
            st.markdown("Accuracy: " + str(result["acc"]))
            st.markdown("Levenshtein distance: " + str(result["levenshtein"]))
            # End of block

if __name__ == "__main__":
    main()
