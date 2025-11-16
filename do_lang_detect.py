import os
import pandas as pd
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from concurrent.futures import ProcessPoolExecutor

languages = ['lt', 'en', 'de']

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_emails(text):
    email_pattern = r'\S+@\S+'
    return re.sub(email_pattern, '', text)

def pick_lang_model(text: str):
    """ Pasirinkti kalbos modeli. """

    if len(text) < 8:
        return None

    try:
        lang = detect(text)
    except (TypeError, LangDetectException):
        return None

    if lang in languages:
        return lang

    else:
        return None

def detect_language_for_text(text: str):
    """Wrapper function that prepares text and returns detected language (or None)."""
    try:
        if not isinstance(text, str) or not text:
            return None
        # Quick length check
        if len(text) < 8:
            return None

        # Clean text
        text = remove_urls(text)
        text = remove_emails(text)

        return pick_lang_model(text)
    except Exception:
        return None

if __name__ == "__main__":
    INPUT_FILE = "daiktai_translated.csv"
    OUTPUT_FILE = "daiktai_translated_with_lang.csv"
    SEP = "ʃ"
    CHUNK_SIZE = 100000

    cpu_count = 6

    # Remove output file if it exists to avoid appending stale data
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    reader = pd.read_csv(
        INPUT_FILE,
        sep=SEP,
        engine="python",
        on_bad_lines='skip',
        chunksize=CHUNK_SIZE,
        dtype=str,
        encoding='utf-8',
    )

    chunk_num = 0
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        for chunk in reader:
            chunk_num += 1
            print(f"Processing chunk {chunk_num}, rows: {len(chunk)}")

            # Fill NaN descriptions with empty string and make sure it's str
            if 'item_description' not in chunk.columns:
                raise KeyError("Expected column 'item_description' not found in CSV")

            # Prepare inputs
            texts = chunk['item_description'].fillna('').astype(str).tolist()

            # Process in parallel. executor.map keeps order consistent with inputs
            results = list(executor.map(detect_language_for_text, texts))

            chunk['detected_lang'] = results

            # Append chunk to output file
            if chunk_num == 1:
                chunk.to_csv(OUTPUT_FILE, sep=SEP, index=False, encoding='utf-8')
            else:
                chunk.to_csv(OUTPUT_FILE, sep=SEP, index=False, header=False, mode='a', encoding='utf-8')

    print(f"Language detection finished. Results saved to {OUTPUT_FILE}")