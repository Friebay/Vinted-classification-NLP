"""
stream_preprocess.py

Memory-efficient CSV preprocessing for large 'daiktai_translated_small_with_lang_emb_ner.csv'
Runs two passes:
  1) Gather classes (Sub_Category_1) and union of ner keys
  2) Transform and write processed rows into an output CSV incrementally.
"""
import ast
import os
import sys
from typing import Set, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Configuration variables
INPUT = "daiktai_translated_small_with_lang_emb_ner.csv"
OUTPUT = "final_df.csv"
CLASSES_OUT = "classes1.npy"
SEP = "ʃ"
CHUNK_SIZE = 100000
EMBED_SIZE = 300
FORCE = False
ENCODING = "utf-8"


# Safely parse a string representation of a Python literal into a Python object.
def safe_literal_eval(s: str):
    if s is None:
        return None
    if isinstance(s, (dict, list)):
        return s
    if isinstance(s, float) and np.isnan(s):
        return None
    # Sometimes values might be "nan" string; handle carefully
    if isinstance(s, str) and s.strip().lower() == "nan":
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        # If we can't parse, silently return None to allow filtering
        return None


def gather_metadata(
    path: str,
    sep: str,
    chunk_size: int,
    columns: list,
    ner_col: str,
    cat_col: str,
    encoding: str = "utf-8",
) -> Tuple[Set[str], Set[str], int]:
    classes = set()
    ner_keys = set()
    total_lines = 0
    kept_lines = 0

    reader = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        on_bad_lines="skip",
        dtype=str,
        encoding=encoding,
        chunksize=chunk_size,
    )

    for chunk in tqdm(reader, desc="Metadata pass", unit="chunk"):
        # Replace 'nan' strings with actual NaN and drop problematic rows
        chunk = chunk.replace("nan", pd.NA)
        # Keep only the columns we need
        chunk = chunk.loc[:, [cat_col, ner_col, "embedding"]].copy()

        total_lines += len(chunk)
        chunk = chunk.dropna(subset=[cat_col, ner_col, "embedding"])
        kept_lines += len(chunk)
        if len(chunk) == 0:
            continue

        # Update classes
        cat_vals = chunk[cat_col].unique().tolist()
        classes.update([v for v in cat_vals if v is not None])

        # Parse ner_dict to collect keys
        for s in chunk[ner_col].values:
            if not isinstance(s, str):
                continue
            d = safe_literal_eval(s)
            if isinstance(d, dict):
                ner_keys.update([k for k in d.keys() if k is not None])

    return classes, ner_keys, total_lines


def process_and_write(
    path: str,
    sep: str,
    chunk_size: int,
    columns: list,
    ner_cols: list,
    class_map: Dict[str, int],
    embed_size: int,
    output_path: str,
    encoding: str = "utf-8",
):
    reader = pd.read_csv(
        path,
        sep=sep,
        engine="python",
        on_bad_lines="skip",
        dtype=str,
        encoding=encoding,
        chunksize=chunk_size,
    )

    header_written = False
    written_rows = 0
    total_read = 0

    embed_col_names = [f"embedding{i}" for i in range(embed_size)]

    # full header for CSV: cat1 first, then embeddings, then ner columns
    header = ["cat1"] + embed_col_names + list(ner_cols)

    for chunk in tqdm(reader, desc="Process pass", unit="chunk"):
        # Keep only columns needed and normalize nan strings
        chunk = chunk.replace("nan", pd.NA)
        chunk = chunk.loc[:, [columns[0], columns[1], "embedding"]].copy()

        total_read += len(chunk)
        chunk = chunk.dropna(subset=[columns[0], columns[1], "embedding"])
        if len(chunk) == 0:
            continue

        # Parse embedding; filter rows with valid embedding
        embeddings_list = []
        keep_mask = []
        for s in chunk["embedding"].values:
            arr = safe_literal_eval(s)
            if (
                isinstance(arr, (list, tuple, np.ndarray))
                and len(arr) == embed_size
            ):
                embeddings_list.append(np.asarray(arr, dtype=np.float32))
                keep_mask.append(True)
            else:
                # invalid embedding size or parse failure -> drop row
                embeddings_list.append(None)
                keep_mask.append(False)

        # Keep only rows with valid embeddings
        keep_mask = np.array(keep_mask, dtype=bool)
        if not keep_mask.any():
            continue

        # Align chunk rows to the valid embedding rows and reset index so everything
        # is 0..n-1 for easy boolean mask operations
        chunk = chunk.iloc[keep_mask].copy()
        chunk.reset_index(drop=True, inplace=True)

        # Build embedding DataFrame (index 0..n-1)
        embeds_arr = np.vstack([a for a in embeddings_list if a is not None])
        df_embeds = pd.DataFrame(embeds_arr, columns=embed_col_names).astype("float32")

        # Parse ner columns and expand to dataframe
        ner_series = chunk[columns[1]].apply(safe_literal_eval)
        df_ner = ner_series.apply(lambda d: pd.Series(d) if isinstance(d, dict) else pd.Series()).fillna(0.0)
        # Ensure all ner columns present; add missing with zeros and reorder to ner_cols list
        missing_cols = set(ner_cols) - set(df_ner.columns)
        for c in missing_cols:
            df_ner[c] = 0.0
        df_ner = df_ner[list(ner_cols)].astype("float32")

        # Create cat1 using class_map and get boolean mask for valid class
        def map_to_cat(val):
            return class_map.get(val, -1)

        df_cat = chunk[columns[0]].map(map_to_cat)

        keep_valid_cat_mask = (df_cat != -1).values  # boolean numpy array for positions 0..n-1
        if keep_valid_cat_mask.sum() == 0:
            continue

        # Filter by mask (works because we reset chunk index earlier)
        df_embeds = df_embeds.iloc[keep_valid_cat_mask].reset_index(drop=True)
        df_ner = df_ner.iloc[keep_valid_cat_mask].reset_index(drop=True)
        df_cat = df_cat.iloc[keep_valid_cat_mask].reset_index(drop=True)

        # Build final chunk
        final_chunk = pd.concat([df_cat.rename("cat1"), df_embeds, df_ner], axis=1)

        # Write the chunk append to CSV
        if not header_written:
            final_chunk.to_csv(output_path, index=False, mode="w")
            header_written = True
        else:
            final_chunk.to_csv(output_path, index=False, mode="a", header=False)

        written_rows += len(final_chunk)

    return total_read, written_rows


def main():
    if not os.path.exists(INPUT):
        print(f"Input file not found: {INPUT}", file=sys.stderr)
        sys.exit(2)

    if os.path.exists(OUTPUT) and not FORCE:
        print(f"Output file {OUTPUT} already exists. Set FORCE = True to overwrite.", file=sys.stderr)
        sys.exit(2)

    columns = ["Sub_Category_1", "ner_dict", "embedding"]
    print("First pass: collecting classes (Sub_Category_1) and union of ner_dict keys.")
    classes, ner_keys, total_lines = gather_metadata(
        INPUT, SEP, CHUNK_SIZE, columns, "ner_dict", "Sub_Category_1", encoding=ENCODING
    )

    classes_sorted = sorted([c for c in classes if c is not None])
    class_map = {c: i for i, c in enumerate(classes_sorted)}
    print(f"Total rows in file (approx): {total_lines}")
    print(f"Found {len(classes_sorted)} classes, {len(ner_keys)} NER keys.")

    # Save classes array
    np.save(CLASSES_OUT, np.array(classes_sorted))
    print(f"Saved classes to {CLASSES_OUT}")

    # Second pass: transform, expand embeddings/ner columns and write to CSV stream
    print("Second pass: processing and writing to CSV (stream).")
    total_read, written_rows = process_and_write(
        INPUT,
        SEP,
        CHUNK_SIZE,
        columns,
        sorted(ner_keys),
        class_map,
        EMBED_SIZE,
        OUTPUT,
        encoding=ENCODING,
    )
    print(f"Total rows read in second pass: {total_read}")
    print(f"Total rows written: {written_rows}")
    print("Done.")


if __name__ == "__main__":
    main()