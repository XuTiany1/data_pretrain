#!/usr/bin/env python3
"""
Fast TinyStories → NLLB translation (one language at a time)
  • FP16 + Flash-Attn
  • Greedy decode (beam=1)
  • Big batches
"""

import os
import argparse
import json
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang",       required=True, help="e.g. aeb_Arab")
    p.add_argument("--data-path",  default="../data/en_Latn_ts.jsonl")
    p.add_argument("--model-dir",  default="/home/mila/x/xut/scratch/model/nllb_model")
    p.add_argument("--out-dir",    default="../data/translations")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1) Load tokenizer + model in FP16 with flash_attn (if installed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"  # requires `pip install flash-attn`
    )
    model.to(device)
    model.eval()

    # 2) Prepare output
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.lang}.jsonl")
    if os.path.exists(out_path):
        print(f"[WARN] {out_path} exists; remove it to re-run.")
        return

    # 3) Read all English stories
    print(f"[INFO] Loading source: {args.data_path}")
    with open(args.data_path) as f:
        data = [json.loads(l) for l in f]

    # 4) Translate in large, FP16-autocast batches
    print(f"[INFO] Translating {len(data)} into {args.lang} → {out_path}")
    with open(out_path, "w") as fout:
        for i in tqdm(range(0, len(data), args.batch_size)):
            batch = data[i : i + args.batch_size]
            src_texts = [ex["text"] for ex in batch]

            enc = tokenizer(
                src_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            forced_id = tokenizer.convert_tokens_to_ids(args.lang)

            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
                out_tokens = model.generate(
                    **enc,
                    forced_bos_token_id=forced_id,
                    max_length=args.max_length,
                    num_beams=1,           # greedy
                    early_stopping=True,
                    use_cache=True,
                )

            decoded = tokenizer.batch_decode(out_tokens, skip_special_tokens=True)
            for src, tgt in zip(src_texts, decoded):
                fout.write(json.dumps({"src": src, "tgt": tgt, "lang": args.lang}, ensure_ascii=False) + "\n")

    print("[DONE]")

if __name__ == "__main__":
    main()
