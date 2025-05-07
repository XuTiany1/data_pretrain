import json
import os
import torch
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import gc
import argparse
from torch.utils.data import Dataset, DataLoader
import time
from functools import partial

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Global configurations
MAX_INPUT_LENGTH = 512  # Reduced from 1024 to 512 to improve memory efficiency
SAVE_CHUNK_SIZE = 10000  # Save after processing this many examples
MEMORY_CHECKPOINT_INTERVAL = 100  # Check memory and clean up every N batches

# Custom dataset for efficient loading
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx]

def collate_fn(batch):
    return batch

def get_memory_usage():
    if torch.cuda.is_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024 ** 3),  # GB
            "cached": torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        }
    return {"allocated": 0, "cached": 0}

def clear_memory():
    """Aggressively clean up memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_model(model_name, use_fp16=True, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load model with optimizations"""
    logger.info(f"Loading model {model_name} on {device}")
    
    # Load tokenizer with caching
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with memory optimization
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Apply performance optimizations
    if device == "cuda":
        model = model.to(device)
        if use_fp16:
            model = model.half()  # Ensure FP16
        
        # Enable further optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    
    model.eval()  # Set to evaluation mode
    
    # Log memory usage after loading
    mem = get_memory_usage()
    logger.info(f"Model loaded. GPU memory allocated: {mem['allocated']:.2f} GB")
    
    return tokenizer, model

def chunked_file_reader(file_path, chunk_size=50000):
    """Read large files in chunks to avoid memory issues"""
    stories = []
    chunk_count = 0
    
    logger.info(f"Reading data in chunks from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                logger.info(f"Read {i} lines so far")
            
            if line.strip():
                try:
                    stories.append(json.loads(line.strip())["text"])
                except json.JSONDecodeError:
                    continue
                    
            if len(stories) >= chunk_size:
                chunk_count += 1
                logger.info(f"Yielding chunk {chunk_count} with {len(stories)} stories")
                yield stories
                stories = []
    
    if stories:  # Don't forget the last chunk
        logger.info(f"Yielding final chunk with {len(stories)} stories")
        yield stories

def translate_chunk(chunk, model, tokenizer, lang_code, batch_size=32, max_length=MAX_INPUT_LENGTH):
    """Translate a chunk of text"""
    logger.info(f"Processing chunk of {len(chunk)} items with batch size {batch_size}")
    
    translated = []
    dataset = TextDataset(chunk)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid fork issues with tokenizers
    )
    
    forced_bos_id = tokenizer.convert_tokens_to_ids(f"__{lang_code}__")
    tokenizer.src_lang = "eng_Latn"
    
    # Process in batches
    for batch_idx, batch_texts in enumerate(tqdm(dataloader, desc=f"Translating batch")):
        if batch_idx % MEMORY_CHECKPOINT_INTERVAL == 0 and batch_idx > 0:
            mem = get_memory_usage()
            logger.info(f"Memory check: {mem['allocated']:.2f} GB allocated, {mem['cached']:.2f} GB cached")
            if mem['allocated'] > 10:  # If using more than 10GB
                logger.info("High memory usage detected, cleaning up...")
                clear_memory()
        
        try:
            # Tokenize the batch
            encoded = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,  # Enable truncation
                max_length=max_length,
            )
            
            # Move to device
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            
            # Generate translations
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=forced_bos_id,
                    max_length=int(max_length * 1.2),
                    num_beams=2,  # Faster beam search
                    early_stopping=True,
                    length_penalty=0.6,  # Slightly penalize length
                )
                
            # Decode translations
            batch_translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            translated.extend(batch_translations)
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"GPU OOM error. Clearing cache and retrying with smaller batch")
                clear_memory()
                
                # Try again with smaller batch size
                if len(batch_texts) > 1:
                    half_point = len(batch_texts) // 2
                    first_half = batch_texts[:half_point]
                    second_half = batch_texts[half_point:]
                    
                    # Process each half separately
                    for mini_batch in [first_half, second_half]:
                        mini_encoded = tokenizer(
                            mini_batch, 
                            return_tensors="pt", 
                            padding=True, 
                            truncation=True,
                            max_length=max_length,
                        )
                        mini_encoded = {k: v.to(model.device) for k, v in mini_encoded.items()}
                        
                        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                            mini_generated = model.generate(
                                **mini_encoded,
                                forced_bos_token_id=forced_bos_id,
                                max_length=int(max_length * 1.2),
                                num_beams=1,  # Use greedy search for recovery
                                early_stopping=True,
                            )
                            
                        mini_translations = tokenizer.batch_decode(mini_generated, skip_special_tokens=True)
                        translated.extend(mini_translations)
                else:
                    # If single sample causes OOM, truncate further
                    logger.warning(f"Single sample causing OOM. Truncating further to {max_length//2}")
                    mini_encoded = tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=max_length//2,
                    )
                    mini_encoded = {k: v.to(model.device) for k, v in mini_encoded.items()}
                    
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                        mini_generated = model.generate(
                            **mini_encoded,
                            forced_bos_token_id=forced_bos_id,
                            max_length=max_length//2,
                            num_beams=1,  # Use greedy search for recovery
                        )
                        
                    mini_translations = tokenizer.batch_decode(mini_generated, skip_special_tokens=True)
                    translated.extend(mini_translations)
            else:
                logger.error(f"Error translating batch: {e}")
                # Add empty strings for this batch to maintain alignment
                translated.extend(["" for _ in range(len(batch_texts))])
                
    return translated

def main():
    parser = argparse.ArgumentParser(description="Optimized NLLB Translation Pipeline")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="Model name")
    parser.add_argument("--input", default="../../../scratch/data/data_pretrain/tinystories/eng_Latn_story.jsonl", help="Input file path")
    parser.add_argument("--output_dir", default="../../../scratch/data/data_pretrain/tinystories", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for translation")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Number of stories to process in each chunk")
    parser.add_argument("--max_length", type=int, default=512, help="Max input length")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 precision")
    parser.add_argument("--languages", nargs="+", default=["fra_Latn", "spa_Latn", "deu_Latn"], help="Target languages")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    tokenizer, model = load_model(args.model, use_fp16=args.fp16)
    
    # Process each language
    for lang_code in args.languages:
        output_path = os.path.join(args.output_dir, f"{lang_code}_story.jsonl")
        
        # Check if we should resume
        if os.path.exists(output_path):
            # Count existing examples
            with open(output_path, "r", encoding="utf-8") as f:
                existing_count = sum(1 for _ in f)
                
            if existing_count > 0:
                logger.info(f"Found existing file with {existing_count} translations. Skipping {lang_code}.")
                continue
        
        logger.info(f"Starting translation to {lang_code}")
        start_time = time.time()
        
        # Process in chunks to manage memory
        chunk_id = 0
        total_translated = 0
        
        with open(output_path, "w", encoding="utf-8") as out_f:
            # Process data in chunks
            for chunk in chunked_file_reader(args.input, chunk_size=args.chunk_size):
                chunk_id += 1
                chunk_time = time.time()
                
                logger.info(f"Processing chunk {chunk_id} ({len(chunk)} examples)")
                translations = translate_chunk(
                    chunk, 
                    model, 
                    tokenizer, 
                    lang_code, 
                    batch_size=args.batch_size,
                    max_length=args.max_length
                )
                
                # Write results
                for t in translations:
                    json.dump({"text": t}, out_f)
                    out_f.write("\n")
                
                # Update counters and log progress
                total_translated += len(translations)
                chunk_elapsed = time.time() - chunk_time
                total_elapsed = time.time() - start_time
                
                # Calculate speeds and ETA
                examples_per_sec = len(translations) / chunk_elapsed
                total_examples_per_sec = total_translated / total_elapsed
                
                logger.info(f"Chunk {chunk_id} completed: {len(translations)} examples in {chunk_elapsed:.2f}s")
                logger.info(f"Speed: {examples_per_sec:.2f} examples/sec for chunk, {total_examples_per_sec:.2f} examples/sec overall")
                
                # Clean up memory after each chunk
                clear_memory()
        
        total_time = time.time() - start_time
        logger.info(f"Completed translation to {lang_code}: {total_translated} examples in {total_time:.2f}s")
        logger.info(f"Average speed: {total_translated/total_time:.2f} examples/sec")
    
    logger.info("All translations completed!")

if __name__ == "__main__":
    main()