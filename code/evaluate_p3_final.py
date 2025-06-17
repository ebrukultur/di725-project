#!/usr/bin/env python
# evaluation_p3_final.py
# Unified few-shot evaluation using sacreBLEU, CIDEr (pycocoevalcap), GeoSim, and sample output logging

import os
import argparse
import pandas as pd
import torch
from PIL import Image
import wandb
from transformers import AutoProcessor
from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration
from peft import PeftModel
from evaluate import load as load_metric
from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.cider.cider import Cider

# Exact prompt used during training
PROMPT = "Describe this scene in English:"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified few-shot evaluation for PaliGemma LoRA model"
    )
    parser.add_argument('--model_dir',      type=str, required=True,
                        help="Path to the fine-tuned PEFT model directory")
    parser.add_argument('--processor_name', type=str, default=None,
                        help="Name or path of the processor (defaults to model_dir)")
    parser.add_argument('--csv',            type=str, required=True,
                        help="Path to the exploded CSV with 'image' and 'caption' columns")
    parser.add_argument('--images_dir',     type=str, required=True,
                        help="Directory containing the images")
    parser.add_argument('--split',          type=str, default='val',
                        help="Which split to evaluate: 'train' or 'val'")
    parser.add_argument('--batch_size',     type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument('--device',         type=str, default='cuda',
                        help="Device for model inference")
    parser.add_argument('--max_length',     type=int, default=64,
                        help="Maximum generation length beyond prompt tokens")
    parser.add_argument('--num_beams',      type=int, default=4,
                        help="Number of beams for beam search")
    parser.add_argument('--max_samples',    type=int, default=None,
                        help="Optional limit on number of samples")
    parser.add_argument('--wandb_project',  type=str, default=None,
                        help="Weights & Biases project (optional)")
    parser.add_argument('--run_name',       type=str, default=None,
                        help="Weights & Biases run name (optional)")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # Load and filter data
    df = pd.read_csv(args.csv)
    df = df[df['split'] == args.split].reset_index(drop=True)
    if args.max_samples:
        df = df.head(args.max_samples)
    grouped = df.groupby('image')['caption'].apply(list).reset_index()
    image_fns = grouped['image'].tolist()
    references = grouped['caption'].tolist()  # list of lists

    # Load processor & model
    # Load processor
    processor = AutoProcessor.from_pretrained(
    args.processor_name or "google/paligemma-3b-pt-224"
    )
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.processor_name or "google/paligemma-3b-pt-224",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(
        base_model, args.model_dir, is_trainable=False
    ) if os.path.isdir(args.model_dir) else base_model
    if isinstance(model, PeftModel):
      print("✅ LoRA adapter loaded and applied.")
    else:
      print("✅ No LoRA adapter loaded — using baseline model.")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    # Prepare metrics
    sacrebleu    = load_metric('sacrebleu')
    cider_scorer = Cider()
    geo_model    = SentenceTransformer('all-mpnet-base-v2', device=device)
    ref_embeds   = [geo_model.encode(caps, convert_to_tensor=True) for caps in references]

    preds = []
    # Batch generation
    for i in range(0, len(image_fns), args.batch_size):
        batch_fns = image_fns[i:i + args.batch_size]
        images = []
        for fn in batch_fns:
            path = os.path.join(args.images_dir, fn)
            try:
                images.append(Image.open(path).convert('RGB'))
            except:
                images.append(Image.new('RGB', (224,224), (0,0,0)))

        enc = processor(
            images=images,
            text=[PROMPT] * len(images),
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_length=enc.input_ids.shape[1] + args.max_length,
                num_beams=args.num_beams,
                early_stopping=True
            )
        decoded = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        preds.extend(decoded)

    # Strip prompt prefix before metrics
    clean_preds = []
    for p in preds:
        if p.startswith(PROMPT):
            clean_preds.append(p[len(PROMPT):].strip())
        else:
            clean_preds.append(p.strip())

    # Compute metrics
    bleu_score = sacrebleu.compute(predictions=clean_preds, references=references)['score']

    cider_refs = {str(idx): references[idx] for idx in range(len(clean_preds))}
    cider_preds = {str(idx): [clean_preds[idx]] for idx in range(len(clean_preds))}
    cider_score, _ = cider_scorer.compute_score(cider_refs, cider_preds)

    pred_embeds = geo_model.encode(clean_preds, convert_to_tensor=True)
    geo_scores = [util.cos_sim(p, r).max().item() for p, r in zip(pred_embeds, ref_embeds)]
    geo_sim = sum(geo_scores) / len(geo_scores)

    # Print metrics
    print(f"SacreBLEU: {bleu_score:.4f}")
    print(f"CIDEr    : {cider_score:.4f}")
    print(f"GeoSim   : {geo_sim:.4f}")

    # Sample Input/Output logging
    print("\n=== Sample Input/Outputs ===")
    for idx in range(min(4, len(image_fns))):
        img_fn = image_fns[idx]
        refs   = references[idx]
        pred   = clean_preds[idx]
        print(f"\nSample {idx+1}:")
        print(f"  Image file: {img_fn}")
        print("  References:")
        for r in refs:
            print(f"    - {r}")
        print(f"  Prediction: {pred}")
        if args.wandb_project:
            try:
                img = Image.open(os.path.join(args.images_dir, img_fn)).convert('RGB')
                caption = f"Refs: {' | '.join(refs)}\nPred: {pred}"
                wandb.log({f'sample_{idx+1}': wandb.Image(img, caption=caption)})
            except:
                pass

    # Log metrics
    if args.wandb_project:
        wandb.log({
            'eval/bleu': bleu_score,
            'eval/cider': cider_score,
            'eval/geo_sim': geo_sim
        })
        wandb.finish()

if __name__ == '__main__':
    main()
