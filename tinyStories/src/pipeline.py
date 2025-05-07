#!/usr/bin/env python3
"""
TinyStories → NLLB translation pipeline using CTranslate2 for speed

Stages:
  download    – fetch dataset + HF model
  translate   – translate with CTranslate2 in parallel
  all         – download then translate
"""

import os
import argparse
import json
import multiprocessing as mp
from datasets import load_dataset, concatenate_datasets
import ctranslate2
from transformers import AutoTokenizer
from tqdm.auto import tqdm

def download_resources(dataset_name: str,
                       data_path: str,
                       hf_model_name: str,
                       hf_model_dir: str):
    # 1. Download / merge TinyStories
    if not os.path.exists(data_path):
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        print(f"Downloading & merging `{dataset_name}` …")
        ds = load_dataset(dataset_name)
        full = concatenate_datasets(list(ds.values()))
        full.to_json(data_path, orient="records", lines=True)
        print(f"Saved English TinyStories to {data_path}")
    else:
        print(f"Found dataset at {data_path}, skipping download.")

    # 2. Download HF NLLB checkpoint (for tokenizer)
    if not (os.path.isdir(hf_model_dir) and os.listdir(hf_model_dir)):
        print(f"Downloading HF model `{hf_model_name}` …")
        tok = AutoTokenizer.from_pretrained(hf_model_name, src_lang="eng_Latn")
        os.makedirs(hf_model_dir, exist_ok=True)
        tok.save_pretrained(hf_model_dir)
        print(f"Saved HF tokenizer to {hf_model_dir}")
    else:
        print(f"Found HF tokenizer in {hf_model_dir}, skipping download.")

def get_target_languages(lang_arg: str):
    if lang_arg.endswith(".json"):
        with open(lang_arg, "r") as f:
            langs = json.load(f)
    else:
        langs = [l.strip() for l in lang_arg.split(",") if l.strip()]
    print(f"🌐 Translating into: {langs}")
    return langs

def translate_one_language(args):
    idx, lang, data_path, out_dir, hf_model_dir, ct2_model_dir, bs, max_len = args

    # Pin this worker to one GPU (if available)
    import torch
    gpu_count = torch.cuda.device_count()
    if gpu_count > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx % gpu_count)

    # Load CTranslate2 translator
    translator = ctranslate2.Translator(
        ct2_model_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="int8",      # must match converter quantization
        inter_threads=1,
        intra_threads=4
    )

    # Load tokenizer (only for detokenization)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, src_lang="eng_Latn")

    # Read English JSONL
    with open(data_path, "r") as f:
        lines = [json.loads(l) for l in f]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang}.jsonl")
    if os.path.exists(out_path):
        print(f"✅ [{lang}] exists, skipping.")
        return

    print(f"🚀 [{lang}] Translating {len(lines)} examples → {out_path}")
    with open(out_path, "w") as fout:
        for i in tqdm(range(0, len(lines), bs), desc=lang):
            batch = lines[i : i + bs]
            texts = [x["text"] for x in batch]

            # CTranslate2 can tokenize for us if we pass raw strings
            results = translator.translate_batch(
                texts,
                target_prefix=[[lang]] * len(texts),  # one-token prefix
                beam_size=1,                          # faster, decent quality
                max_batch_size=bs,
                length_penalty=1.0,
                remove_bpe=True
            )

            # Detokenize hypotheses
            for src, res in zip(texts, results):
                hyp_tokens = res.hypotheses[0]
                # HF tokenizer can convert list-of-tokens → string
                pred = tokenizer.convert_tokens_to_string(hyp_tokens)
                fout.write(json.dumps({
                    "src": src,
                    "tgt": pred,
                    "lang": lang
                }, ensure_ascii=False) + "\n")

    print(f"✅ [{lang}] done.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",    choices=["download","translate","all"], default="all")
    p.add_argument("--dataset",  default="roneneldan/TinyStories")
    p.add_argument("--data-path", default="../data/en_Latn_ts.jsonl")
    p.add_argument("--hf-model-name", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--hf-model-dir",  default="/home/mila/x/xut/scratch/model/nllb-hf-tokenizer")
    p.add_argument("--ct2-model-dir", default="/home/mila/x/xut/scratch/model/nllb_model-ct2")
    p.add_argument("--langs",     default="aeb_Arab,fra_Latn")
    p.add_argument("--out-dir",   default="../data/translations")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()

    if args.stage in ("all","download"):
        download_resources(
            args.dataset,
            args.data_path,
            args.hf_model_name,
            args.hf_model_dir
        )

    if args.stage in ("all","translate"):
        langs = get_target_languages(args.langs)
        jobs = [
            (
              i, lang,
              args.data_path,
              args.out_dir,
              args.hf_model_dir,
              args.ct2_model_dir,
              args.batch_size,
              args.max_length
            )
            for i, lang in enumerate(langs)
        ]
        with mp.Pool(len(jobs)) as pool:
            pool.map(translate_one_language, jobs)

if __name__ == "__main__":
    main()
