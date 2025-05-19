# Import libraries
import os
import argparse
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
import wandb

#RISC Dataset class
class RiscCaptionDataset(Dataset):
    def __init__(self, csv_path, images_dir, processor, max_len=50, split="train"):
        # Load csv and filter by the specified split
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor
        self.max_len = max_len

    def __len__(self):
        # total number of samples in this split
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load image
        img_path = os.path.join(self.images_dir, row["image"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224,224), (255,255,255))
        # Encoder inputs
        enc = self.processor(images=image, return_tensors="pt")
        pixel_values = enc.pixel_values.squeeze(0)
        # *** NEW: prefix with exactly one <image> token ***
        syn = row["syn_tokens"]
        tokens = ast.literal_eval(syn) if isinstance(syn, str) else syn
        text = "<image> " + " ".join(tokens)
        # processor to tokenize both image and text
        proc = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )
        # extract tensors, remove batch dimension
        pixel_values = proc.pixel_values.squeeze(0)
        input_ids = proc.input_ids.squeeze(0)
        attention_mask = proc.attention_mask.squeeze(0)
        # Labels
        labels = input_ids.clone()
        pad_id  = self.processor.tokenizer.pad_token_id
        img_id  = self.processor.tokenizer.convert_tokens_to_ids("<image>")
        labels[labels == pad_id]  = -100
        labels[labels == img_id]  = -100

        # item returns
        return {
            "pixel_values":   pixel_values, #preprocessed image tensor for the model's vision encoder
            "input_ids":      input_ids, #token IDs for the caption sequence
            "attention_mask": attention_mask, #attention mask corresponding to input_ids
            "labels":         labels #pad and image tokens masked
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="di725-project")
    parser.add_argument("--run_name", type=str, default="lora_finetune")
    args = parser.parse_args()

    # Initialize W&B with config
    wandb.login()
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )

    # Load processor & base model
    processor = AutoProcessor.from_pretrained(args.model_name)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16
    )
    # Prepare LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base_model, peft_config)

    # Prepare datasets
    train_ds = RiscCaptionDataset(args.csv, args.images_dir, processor, split="train")
    val_ds   = RiscCaptionDataset(args.csv, args.images_dir, processor, split="val")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        do_eval=True,
        eval_steps=500,
        fp16=True,
        report_to=["wandb"]
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        tokenizer=processor.tokenizer
    )

    # Train & save
    trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    processor.save_pretrained(os.path.join(args.output_dir, "processor"))

    # Final evaluation
    metrics = trainer.evaluate()
    print(metrics)
    wandb.log(metrics)

if __name__ == "__main__":
    main()

