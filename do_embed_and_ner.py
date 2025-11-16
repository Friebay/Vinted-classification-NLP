import os
import pandas as pd
import re
import json
import spacy
import time
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

# spaCy model names for each language
SPACY_MODELS = {
    'lt': 'lt_core_news_lg',
    'en': 'en_core_web_lg',
    'de': 'de_core_news_lg',
}


def get_ner_dict(doc):
    output = {
        'numeric': 0,
        'currency': 0
    }

    # spacy NER
    for ent in doc.ents:
        if ent.label_ not in output:
            output[ent.label_] = 1
        else:
            output[ent.label_] += 1

    for tok in doc:
        # numerical
        if tok.is_digit:
            output['numeric'] += 1

        # currency
        if tok.is_currency:
            output['currency'] += tok.is_currency

    return output


def process_language_batch(df_lang, lang_code, model_name, output_file, sep, write_header):
    print(f"  Loading spaCy model: {model_name} for language: {lang_code}")
    try:
        nlp = spacy.load(model_name)
        # Disable unnecessary pipeline components for speed
        disabled_pipes = []
        for pipe_name in nlp.pipe_names:
            if pipe_name not in ['tok2vec', 'transformer', 'ner', 'tagger']:
                disabled_pipes.append(pipe_name)
        
        if disabled_pipes:
            print(f"  Disabled pipes for speed: {disabled_pipes}")
            nlp.disable_pipes(*disabled_pipes)
        
        # Set smaller max_length to prevent huge docs
        nlp.max_length = 1000000  # Adjust as needed
            
    except OSError:
        print(f"  WARNING: Model {model_name} not found. Please install with: python -m spacy download {model_name}")
        df_lang['ner_dict'] = None
        df_lang['embedding'] = None
        df_lang.to_csv(output_file, sep=sep, index=False, encoding='utf-8', 
                      mode='a', header=write_header)
        return len(df_lang)
    
    total = len(df_lang)
    print(f"  Processing {total} rows for language: {lang_code}")
    
    texts = df_lang['item_description'].fillna('').astype(str).tolist()
    
    # Process in chunks to avoid memory buildup
    CHUNK_SIZE = 10000  # Process and write every 10k rows
    batch_size = 250
    
    start_time = time.time()
    progress_interval = max(1, total // 100)
    
    processed = 0
    chunk_start = 0
    
    # Process in chunks
    while chunk_start < total:
        chunk_end = min(chunk_start + CHUNK_SIZE, total)
        chunk_texts = texts[chunk_start:chunk_end]
        chunk_indices = df_lang.index[chunk_start:chunk_end]
        
        # Process this chunk
        ner_dicts = []
        embeddings = []
        
        for i, doc in enumerate(nlp.pipe(chunk_texts, batch_size=batch_size, n_process=1), start=1):
            ner_dict = get_ner_dict(doc)
            embedding = doc.vector.tolist()
            
            ner_dicts.append(json.dumps(ner_dict) if ner_dict else None)
            embeddings.append(json.dumps(embedding) if embedding else None)
            
            processed = chunk_start + i
            if (processed % progress_interval) == 0 or processed == total:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"    [{lang_code}] Processed {processed}/{total} ({processed/total:.1%}) rows — {rate:.1f} rows/sec")
        
        # Create chunk dataframe and write immediately
        df_chunk = df_lang.loc[chunk_indices].copy()
        df_chunk['ner_dict'] = ner_dicts
        df_chunk['embedding'] = embeddings
        
        # Write to CSV (append mode)
        mode = 'w' if write_header and chunk_start == 0 else 'a'
        header = write_header and chunk_start == 0
        df_chunk.to_csv(output_file, sep=sep, index=False, encoding='utf-8',
                       mode=mode, header=header)
        
        # Clear memory
        del df_chunk, ner_dicts, embeddings, chunk_texts
        gc.collect()
        
        chunk_start = chunk_end
    
    elapsed_total = time.time() - start_time
    avg_rate = processed / elapsed_total if elapsed_total > 0 else 0
    print(f"  Finished language {lang_code}: {processed}/{total} rows, total time {elapsed_total:.1f}s, avg {avg_rate:.1f} rows/sec")
    
    return processed


if __name__ == "__main__":
    INPUT_FILE = "daiktai_translated_small_with_lang.csv"
    OUTPUT_FILE = "daiktai_translated_small_with_lang_emb_ner.csv"
    SEP = "ʃ"
    
    # Language processing order
    LANGUAGE_ORDER = ['lt', 'en', 'de']

    # Remove output file if it exists
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    print(f"Reading input file: {INPUT_FILE}")
    df = pd.read_csv(
        INPUT_FILE,
        sep=SEP,
        engine="python",
        on_bad_lines='skip',
        dtype=str,
        encoding='utf-8',
    )
    
    print(f"Total rows loaded: {len(df)}")
    
    if 'detected_lang' not in df.columns:
        raise KeyError("Expected column 'detected_lang' not found in CSV. Please run do_lang_detect.py first.")
    
    cumulative_processed = 0
    first_write = True
    
    for lang_code in LANGUAGE_ORDER:
        df_lang = df[df['detected_lang'] == lang_code].copy()
        
        if len(df_lang) == 0:
            print(f"No rows found for language: {lang_code}, skipping...")
            continue
        
        print(f"\nProcessing language: {lang_code} ({len(df_lang)} rows)")
        
        model_name = SPACY_MODELS.get(lang_code)
        if not model_name:
            print(f"  No spaCy model configured for language: {lang_code}, skipping...")
            continue
        
        # Process and write directly to file
        rows_processed = process_language_batch(
            df_lang, lang_code, model_name, OUTPUT_FILE, SEP, first_write
        )
        first_write = False
        cumulative_processed += rows_processed
        
        # Clear memory
        del df_lang
        gc.collect()
        
        print(f"  Cumulative rows processed so far: {cumulative_processed}")
    
    # Handle rows with unrecognized languages
    df_other = df[~df['detected_lang'].isin(LANGUAGE_ORDER)].copy()
    if len(df_other) > 0:
        print(f"\nProcessing {len(df_other)} rows with unrecognized/missing language...")
        df_other['ner_dict'] = None
        df_other['embedding'] = None
        df_other.to_csv(OUTPUT_FILE, sep=SEP, index=False, encoding='utf-8',
                       mode='a', header=first_write)
        cumulative_processed += len(df_other)
        print(f"  Cumulative rows processed so far: {cumulative_processed}")
    
    print(f"\nProcessing complete! Results saved to {OUTPUT_FILE}")
    print(f"Total rows processed: {cumulative_processed}")