"""
train_lora.py

Phase 2 LoRA fine-tuning on RISC dataset using PaliGemma.
 - Load processed captions CSV
 - Define PyTorch Dataset and DataLoader
 - Initialize PaliGemma with LoRA adapters (PEFT)
 - Train with HuggingFace Trainer, log to W&B
 - Evaluate BLEU-4 and CIDEr on validation split
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from transformers import AutoProcessor, TrainingArguments, VisionEncoderDecoderModel, Trainer
from transformers import DataCollatorForSeq2Seq

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import wandb

class RiscCaptionDataset(Dataset):
    def __init__(self, csv_path, images_dir, processor, split):
        self.df = pd.read_csv(csv_path)
        # filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.images_dir, row['image'])).convert('RGB')
        text_tokens = " ".join(row['syn_tokens'])

        # prepare inputs for encoder+decoder
        proc_outputs = self.processor(
             images=image,
             text=text_tokens,
             return_tensors='pt',
             padding='max_length',
             truncation=True
         )
        # proc_outputs contains:
        #   pixel_values, input_ids, attention_mask

        # Trainer needs a `labels` tensor to compute cross‐entropy loss
        proc_outputs['labels'] = proc_outputs['input_ids'].clone()

        # squeeze off the batch dimension
        return {k: v.squeeze(0) for k, v in proc_outputs.items()}


    def get_split(self):
        return self.df['split'].iloc[0]


def main(args):
    # Initialize W&B
    wandb.login()
    run = wandb.init(project=args.wandb_project, name=args.run_name)

    # Processor and model
    #processor = AutoProcessor.from_pretrained(args.model_name)
    #base_model = AutoModelForCausalLM.from_pretrained(args.model_name)

    processor = AutoProcessor.from_pretrained(args.model_name, use_auth_token=True)
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(args.model_name,use_auth_token=True, torch_dtype=torch.bfloat16)
    for name, module in base_model.named_modules():
      if isinstance(module, torch.nn.Linear):
        print(name)


    # Prepare LoRA config
    peft_config = LoraConfig(
        #task_type=TaskType.CAUSAL_LM,
        #task_type=TaskType.VISION_ENCODER_DECODER,
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(base_model, peft_config)


    # Datasets & DataLoaders
    train_dataset = RiscCaptionDataset(args.csv, args.images_dir, processor, split='train')
    val_dataset   = RiscCaptionDataset(args.csv, args.images_dir, processor, split='val')

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        #gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        logging_steps=100,
        save_steps=500,
        learning_rate=args.lr,
        #weight_decay=0.01,
        fp16=True,
        warmup_steps=500,
        #max_grad_norm=1.0,
        report_to=['wandb']
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        label_pad_token_id=processor.tokenizer.pad_token_id,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.tokenizer,
        data_collator=data_collator
    )

    # Train
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    model.save_pretrained(os.path.join(args.output_dir, "lora_adapter"))
    processor.save_pretrained(os.path.join(args.output_dir, "processor"))
    # Evaluate
    metrics = trainer.evaluate()
    print(metrics)
    wandb.log(metrics)
    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='/content/drive/MyDrive/DI 725 Project/data/captions_processed.csv')
    parser.add_argument('--images_dir', type=str, default='/content/drive/MyDrive/DI 725 Project/data/images')
    parser.add_argument('--model_name', type=str, default='google/paligemma2-28b-mix-224')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--wandb_project', type=str, default='risc-image-captioning')
    parser.add_argument('--run_name', type=str, default='lora_train')
    parser.add_argument('--train_bs', type=int, default=16)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="Path to a checkpoint directory to resume training from")

    args = parser.parse_args()
    main(args)