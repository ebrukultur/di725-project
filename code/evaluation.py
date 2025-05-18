# evaluation4.py

import os
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt

import wandb
from transformers import AutoProcessor
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from peft import PeftModel
import evaluate
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from sentence_transformers import SentenceTransformer, util

# ─── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/drive/MyDrive/DI 725 Project/out/lora_r4"
CSV_PATH   = "/content/drive/MyDrive/DI 725 Project/data/captions.csv"
IMAGES_DIR = "/content/drive/MyDrive/DI 725 Project/data/resized"
BATCH_SIZE = 8
MAX_LEN    = 50
# ────────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Initialize W&B
    wandb.login()
    run = wandb.init(
        project="di725-project",
        name="eval-lora-r4",
        config={"output_dir": OUTPUT_DIR, "batch_size": BATCH_SIZE, "max_length": MAX_LEN}
    )

    # 2) Load CSV & keep only val split
    df = pd.read_csv(CSV_PATH)
    df = df[df["split"] == "val"].reset_index(drop=True)

    # 3) Group the 5 captions per image and flatten
    grouped = df.groupby("image")[
        ["caption_1","caption_2","caption_3","caption_4","caption_5"]
    ].agg(list).reset_index()

    images = grouped["image"].tolist()
    # flatten into list of strings per image
    references = grouped.apply(
        lambda row: [
            cap for sub in [
                row.caption_1, row.caption_2, row.caption_3,
                row.caption_4, row.caption_5
            ] for cap in sub if isinstance(cap, str) and cap.strip()
        ],
        axis=1
    ).tolist()

    # 4) Load processor & LoRA-fine-tuned model
    processor = AutoProcessor.from_pretrained(os.path.join(OUTPUT_DIR, "processor"))
    base = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma2-3b-pt-224", torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, os.path.join(OUTPUT_DIR, "lora_adapter"))
    model.eval().to("cuda")

    # 5) Prepare metrics
    bleu_metric   = evaluate.load("bleu")
    cider_scorer  = Cider()
    ptb           = PTBTokenizer()
    geo_model     = SentenceTransformer('all-mpnet-base-v2', device="cuda")

    all_preds, all_refs = [], []

    # 6) Inference loop (true zero-shot: only images + "<image>" tokens)
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i : i + BATCH_SIZE]
        batch_refs = references[i : i + BATCH_SIZE]

        pil_images = []
        for fn in batch_imgs:
            try:
                img = Image.open(os.path.join(IMAGES_DIR, fn)).convert("RGB")
            except:
                img = Image.new("RGB", (224,224), (255,255,255))
            pil_images.append(img)

        # **Key**: one "<image>" token per image
        prompts = ["<image>"] * len(pil_images)

        inputs = processor(
            images=pil_images,
            text=prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        ).to("cuda")

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_LEN,
                pad_token_id=processor.tokenizer.pad_token_id
            )
        preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

        all_preds.extend(preds)
        all_refs.extend(batch_refs)

    # 7) Compute BLEU-4
    bleu_score = bleu_metric.compute(predictions=all_preds, references=all_refs)["bleu"]

    # 8) Compute CIDEr
    coco_refs  = {i: [{"caption": c}] for i, caps in enumerate(all_refs) for c in caps}
    coco_preds = {i: [{"caption": p}] for i, p in enumerate(all_preds)}
    refs_tok   = ptb.tokenize(coco_refs)
    preds_tok  = ptb.tokenize(coco_preds)
    cider_score, _ = cider_scorer.compute_score(refs_tok, preds_tok)

    # 9) Compute Geo-Sim
    flat_refs = [" ".join(caps) for caps in all_refs]
    emb_refs  = geo_model.encode(flat_refs,  convert_to_tensor=True)
    emb_preds = geo_model.encode(all_preds, convert_to_tensor=True)
    geo_score = util.cos_sim(emb_preds, emb_refs).diag().mean().item()

    # 10) Log & print metrics
    wandb.log({
        "eval/bleu": bleu_score,
        "eval/cider": cider_score,
        "eval/geo_sim": geo_score
    })
    print("=== Evaluation Metrics ===")
    print(f"BLEU-4 : {bleu_score:.4f}")
    print(f"CIDEr  : {cider_score:.4f}")
    print(f"GeoSim : {geo_score:.4f}")

    # 11) Show 4 sample images
    for idx in range(min(4, len(images))):
        img = Image.open(os.path.join(IMAGES_DIR, images[idx])).convert("RGB")
        caps = all_refs[idx]
        pred = all_preds[idx]
        title = "GT:\n" + "\n".join(caps) + "\n\nPRED:\n" + pred

        plt.figure(figsize=(4,4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(title, wrap=True)
        plt.show()
        wandb.log({f"sample_{idx}": wandb.Image(img, caption=title)})

    run.finish()

if __name__ == "__main__":
    main()
