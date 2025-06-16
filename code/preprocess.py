"""
preprocess.py

Phase 2 preprocessing for RISC dataset:
 - Unify multiple caption columns into long format
 - Filter captions by length
 - Normalize text
 - Train/apply SentencePiece for tokenization
 - Cluster tokens via DBSCAN
 - Map synonyms using WordNet
 - Output cleaned captions_processed.csv
"""

import os
import re
import argparse
import pandas as pd
import sentencepiece as spm
from sklearn.cluster import DBSCAN
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import nltk

# Ensure NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')


def load_captions(csv_path):
    # auto-detect delimiter
    df = pd.read_csv(csv_path, sep=None, engine='python')
    df.columns = df.columns.str.strip().str.lower()
    # melt caption_1...caption_5 into long form
    caption_cols = [c for c in df.columns if c.startswith('caption_')]
    df_long = df.melt(
        id_vars=[c for c in df.columns if not c.startswith('caption_')],
        value_vars=caption_cols,
        var_name='caption_id',
        value_name='caption'
    )
    # drop missing/empty
        # drop missing/empty and duplicate rows
    df_long = df_long.dropna(subset=['caption'])
    df_long = df_long.drop_duplicates(subset=['image','caption_id','caption'])
    df_long = df_long[df_long['caption'].str.strip().astype(bool)]
    return df_long


def filter_by_length(df, min_len=5, max_len=20):
    df['length'] = df['caption'].str.split().apply(len)
    out = df[(df['length'] >= min_len) & (df['length'] <= max_len)].copy()
    out.drop(columns='length', inplace=True)
    return out


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text


def train_sentencepiece(captions, model_prefix='risc_spm', vocab_size=2000):
    with open('all_captions.txt', 'w', encoding='utf8') as f:
        for cap in captions:
            f.write(cap + '\n')
    spm.SentencePieceTrainer.Train(
        f"--input=all_captions.txt --model_prefix={model_prefix} --vocab_size={vocab_size}"
    )


def load_sp_model(model_prefix='risc_spm.model'):
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix)
    return sp


def tokenize_captions(df, sp):
    df['tokens'] = df['caption'].apply(lambda x: sp.encode(x, out_type=str))
    return df


def cluster_tokens(df, eps=0.5, min_samples=5):
    all_tokens = list({t for toks in df['tokens'] for t in toks})
    import numpy as np
    X = np.array([[len(t), sum(ord(c) for c in t)] for t in all_tokens])
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    token2cluster = dict(zip(all_tokens, db.labels_))
    df['clustered_tokens'] = df['tokens'].apply(lambda toks: [str(token2cluster[t]) for t in toks])
    return df, token2cluster


def map_synonyms(df):
    def syn_map(token):
        syns = wordnet.synsets(token)
        return syns[0].lemmas()[0].name() if syns else token
    df['syn_tokens'] = df['clustered_tokens'].apply(lambda toks: [syn_map(t) for t in toks])
    return df


def save_processed(df, output_csv):
    df[['image', 'caption_id', 'syn_tokens', 'split']].to_csv(output_csv, index=False)


def main(args):
    df = load_captions(args.input_csv)
    df = filter_by_length(df, args.min_len, args.max_len)
    df['caption'] = df['caption'].apply(normalize_text)

    if not os.path.exists(args.spm_model):
        train_sentencepiece(df['caption'], model_prefix=args.spm_prefix, vocab_size=args.vocab_size)
    sp = load_sp_model(args.spm_model)

    df = tokenize_captions(df, sp)
    df, token2cluster = cluster_tokens(df, args.eps, args.min_samples)
    df = map_synonyms(df)
    save_processed(df, args.output_csv)
    print(f"Processed captions saved to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="/content/drive/MyDrive/DI 725 Project/data/captions.csv")
    parser.add_argument("--output_csv", type=str, default="/content/drive/MyDrive/DI 725 Project/data/captions_processed.csv")
    parser.add_argument("--min_len", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--spm_prefix", type=str, default="risc_spm")
    parser.add_argument("--spm_model", type=str, default="risc_spm.model")
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    args = parser.parse_args()
    main(args)
    main(args)
