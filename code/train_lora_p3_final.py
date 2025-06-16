#!/usr/bin/env python
# train_lora_p3_final.py
# LoRA fine-tuning with a single prompt

import argparse
import os
import torch
import pandas as pd
import multiprocessing
import wandb
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.integrations import WandbCallback
import peft

# single-shot prompt (handled by processor)
PROMPT = "Describe this scene in English:"

def parse_args():
    p = argparse.ArgumentParser(description="Few-shot LoRA fine-tuning for PaliGemma")
    p.add_argument('--model_name',    type=str, default='google/paligemma-3b-pt-224')
    p.add_argument('--csv',           type=str, required=True)
    p.add_argument('--images_dir',    type=str, required=True)
    p.add_argument('--output_dir',    type=str, required=True)
    p.add_argument('--wandb_project', type=str, default=None)
    p.add_argument('--run_name',      type=str, default=None)
    p.add_argument('--fp16',          action='store_true')
    p.add_argument('--max_samples','--max_train_samples', type=int, default=None)
    p.add_argument('--train_bs',      type=int, default=4)
    p.add_argument('--eval_bs',       type=int, default=4)
    p.add_argument('--epochs',        type=int, default=3)
    p.add_argument('--lr',            type=float, default=5e-7)
    p.add_argument('--lora_r',        type=int, default=8)
    p.add_argument('--lora_alpha',    type=int, default=16)
    p.add_argument('--lora_dropout',  type=float, default=0.1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # load & split
    df = pd.read_csv(args.csv)
    ds = Dataset.from_pandas(df)
    train_ds = ds.filter(lambda x: x['split']=='train')
    val_ds   = ds.filter(lambda x: x['split']=='val')
    if args.max_samples:
        train_ds = train_ds.select(range(min(len(train_ds), args.max_samples)))

    # processor & model
    processor = AutoProcessor.from_pretrained(args.model_name)
    dtype = torch.float16 if args.fp16 else None
    if 'paligemma' in args.model_name.lower():
        backbone = PaliGemmaForConditionalGeneration.from_pretrained(
            args.model_name, torch_dtype=dtype
        )
    else:
        backbone = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name, torch_dtype=dtype
        )

    # LoRA config
    peft_conf = peft.LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='SEQ_2_SEQ_LM',
        target_modules=['q_proj','v_proj']
    )
    model = peft.get_peft_model(backbone, peft_conf)
    print("⚙️  Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.cuda().train()

    # closure-based preprocessing function
    def preprocess_fn(batch):
        # load images
        images = [
            Image.open(os.path.join(args.images_dir, fn)).convert("RGB")
            for fn in batch['image']
        ]
        # encode prompt + caption
        enc = processor(
            images=images,
            text=[f"{PROMPT} {cap}" for cap in batch['caption']],
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        # clone input_ids to labels
        labels = enc.input_ids.clone()
        # mask out prompt prefix tokens
        prompt_ids = processor.tokenizer(
            PROMPT, add_special_tokens=False
        ).input_ids
        labels[:, :len(prompt_ids)] = -100
        # mask padding tokens
        labels[labels == processor.tokenizer.pad_token_id] = -100
        enc['labels'] = labels
        return enc

    # apply preprocessing
    train_ds = train_ds.with_transform(preprocess_fn)
    val_ds   = val_ds.with_transform(preprocess_fn)

    # data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    # BLEU metric
    def compute_metrics(pred):
        from datasets import load_metric
        bleu = load_metric('bleu')
        dp = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        dl = processor.tokenizer.batch_decode(pred.label_ids,     skip_special_tokens=True)
        p  = [x.split() for x in dp]
        l  = [[y.split()] for y in dl]
        return {'bleu': bleu.compute(predictions=p, references=l)['bleu']}

    num_workers = min(multiprocessing.cpu_count(), 4) #multiprocessing to 4 cores

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=800,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,  # clip gradients
        num_train_epochs=args.epochs,
        logging_steps=100,
        dataloader_num_workers=num_workers,
        report_to=['wandb'] if args.wandb_project else [],
        run_name=args.run_name,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[WandbCallback()]
    )

    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()
