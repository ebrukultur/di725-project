"""
preprocess_p3.py

Preprocess RISC dataset captions for LoRA Phase 3.
- Load captions.csv (one row per image with 5 captions)
- Shuffle images
- Split images into 80/10/10 fractions (train/validation/test)
- Explode each image’s 5 captions into separate rows
- Shuffle caption rows and save combined CSV
"""

import pandas as pd
import argparse


def preprocess_phase3(input_csv: str,
                      output_csv: str,
                      seed: int = 42) -> None:
    # Load the CSV of images with 5 captions each
    df = pd.read_csv(input_csv)

    # Shuffle at image level
    df_shuf = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    total_images = len(df_shuf)

    # 80/10/10 split
    train_n = int(0.8 * total_images)
    val_n   = int(0.1 * total_images)
    test_n  = total_images - train_n - val_n

    df_train_img = df_shuf.iloc[:train_n]
    df_val_img   = df_shuf.iloc[train_n:train_n + val_n]
    df_test_img  = df_shuf.iloc[train_n + val_n:]

    # Explode each subset into (image, single-caption) rows
    def explode(df_subset, split_name):
        rows = []
        caption_cols = ['caption_1','caption_2','caption_3','caption_4','caption_5']
        for _, row in df_subset.iterrows():
            for col in caption_cols:
                cap = str(row.get(col, '')).strip()
                if cap:
                    rows.append({'image': row['image'], 'split': split_name, 'caption': cap})
        return pd.DataFrame(rows)

    train_df = explode(df_train_img, 'train')
    val_df   = explode(df_val_img, 'val')
    test_df  = explode(df_test_img, 'test')

    # Combine and shuffle all caption rows
    out_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    out_df = out_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save to CSV
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test caption rows to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess RISC dataset captions for LoRA Phase 3"
    )
    parser.add_argument('--input_csv',  type=str, required=True,
                        help="Path to the input captions CSV (one row per image)")
    parser.add_argument('--output_csv', type=str, required=True,
                        help="Path to save processed CSV with exploded captions")
    parser.add_argument('--seed',       type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    preprocess_phase3(
        args.input_csv,
        args.output_csv,
        args.seed
    )
