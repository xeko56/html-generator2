import os
import signal
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from callback import PushToHubCallback
from dataset import HtmlTablesDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import DonutModelPLModule

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

image_size = [1280, 960]
max_length = 768

def handle_interrupt(sig, frame, model):
    print("Training interrupted. Saving the model...")
    model.save_pretrained(save_directory="./save_pretrained")
    print("Model saved. Exiting...")
    exit(0)

def main():
    config = VisionEncoderDecoderConfig.from_pretrained("naver-clova-ix/donut-base")
    config.encoder.image_size = image_size
    config.decoder.max_length = max_length

    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base", config=config)

    added_tokens = []

    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis = False

    tokenizer = processor.tokenizer

    # Initialize the dataset

    train_dataset = HtmlTablesDataset(
        processor=processor,
        json_file='./data_pairs.json',
        tokenizer=tokenizer,
        model=model, 
        max_length=max_length,                         
        split="train", 
        task_start_token="<html_table>", 
        prompt_end_token="<html_table>",
        ignore_id=-100,
        added_tokens=added_tokens,
    )

    val_dataset = HtmlTablesDataset(
        processor=processor,
        json_file='./data_pairs.json',
        tokenizer=tokenizer,
        model=model,
        max_length=max_length,                         
        split="validation", 
        task_start_token="<html_table>", 
        prompt_end_token="<html_table>",
        ignore_id=-100,
        added_tokens=added_tokens,
    )

    # pixel_values, labels, target_sequence = train_dataset[0]
    # print(target_sequence)
    # print(labels)
    # # for id in labels.tolist()[:30]:
    # #     if id != -100:
    # #         print(processor.decode([id]))
    # #     else:
    # #         print(id)
    # return

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<html_table>'])[0]

    # Initialize the model
    config = {
        "max_epochs":1,
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
  
    model_module = DonutModelPLModule(config, processor, model, train_dataset, val_dataset, max_length)

    checkpoint_dir = "./model_checkpoints"
    last_checkpoint = os.path.join(checkpoint_dir, "last-v3.ckpt")

    if os.path.exists(last_checkpoint):
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        model_module = DonutModelPLModule.load_from_checkpoint(last_checkpoint, config=config, processor=processor, model=model, train_dataset=train_dataset, val_dataset=val_dataset, max_length=max_length)

    signal.signal(signal.SIGINT, lambda sig, frame: handle_interrupt(sig, frame, model_module.model))

    # Initialize the logger
    logger = WandbLogger(project="HTML Generator", name="html-generator-level-2")

    early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

    checkpoint_callback = ModelCheckpoint(
        dirpath="./model_checkpoints",  # Directory to save the model
        filename="{epoch}-{val_edit_distance:.2f}",  # File naming convention
        monitor="val_edit_distance",  # Metric to monitor for saving the best model
        mode="min",  # Save model with minimum val_edit_distance
        save_top_k=3,  # Save top 3 models based on monitored metric
        save_last=True  # Additionally, save the last model state at the end of training
    )

    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=config.get("max_epochs"),
            val_check_interval=config.get("val_check_interval"),
            check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
            gradient_clip_val=config.get("gradient_clip_val"),
            precision="16-mixed",
            num_sanity_val_steps=0,
            logger=logger,
            callbacks=[PushToHubCallback(), checkpoint_callback, early_stop_callback],
            fast_dev_run=False,
    )

    trainer.fit(model_module)       


if __name__ == "__main__":
    main()
