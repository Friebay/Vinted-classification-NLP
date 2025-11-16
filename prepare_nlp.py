import os
import re

from joblib import Parallel, delayed
import spacy

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

spacy_models = {
    'lt': spacy.load('lt_core_news_sm'),
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm'),
}

# https://github.com/explosion/spaCy/discussions/9147
ALL_ENTITY_TYPES = [
    "CARDINAL", 
    "DATE", 
    "EVENT", 
    "FAC", 
    "GPE", 
    "LANGUAGE", 
    "LAW", 
    "LOC", 
    "MONEY", 
    "NORP", 
    "ORDINAL", 
    "ORG", 
    "PERCENT", 
    "PERSON", 
    "PRODUCT", 
    "QUANTITY", 
    "TIME", 
    "WORK_OF_ART",
    "LOC", 
    "MISC", 
    "ORG", 
    "PER"
]


def pick_lang_model(text: str, spacy_models):
    """ Pasirinkti kalbos modeli. """

    if len(text) < 8:
        return None

    try:
        lang = detect(text)
    except (TypeError, LangDetectException):
        return None

    if lang in spacy_models:
        return spacy_models[lang]

    else:
        return None


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_emails(text):
    email_pattern = r'\S+@\S+'
    return re.sub(email_pattern, '', text)


def get_ner_dict(doc):
    
    # Inicializuoti output dictionary su visais NER type'ais
    output = {
        ent: 0
        for ent in ALL_ENTITY_TYPES
    }

    output['numeric'] = 0
    output['currency'] = 0

    # spacy NER
    for ent in doc.ents:
        output[ent.label_] += 1 / len(doc)

    for tok in doc:
        # numerical
        if tok.is_digit:
            output['numeric'] += 1 / len(doc)

        # currency
        if tok.is_currency:
            output['currency'] += 1 / len(doc)

    return output


def parse_and_count_entities(data_row: dict):
    # Prepare data
    text = data_row['item_description']

    text = remove_urls(text)
    text = remove_emails(text)

    spacy_model = pick_lang_model(text, spacy_models)

    # Jei nepagavo vienos is top 3 kalbu
    if spacy_model is None:
        data_row['ner_dict'] = None
        data_row['embedding'] = None

        return data_row

    doc = spacy_model(text)
    data_row['ner_dict'] = get_ner_dict(doc)
    data_row['embedding'] = doc.vector

    return data_row
if __name__ == "__main__":
    df = pd.read_csv(
        "daiktai_translated_small.csv",
        sep="ʃ",
        engine="python",
        on_bad_lines='skip'
    )
    df_records = df.to_dict('records')

    with Parallel(
        n_jobs=os.cpu_count() - 2,
        mmap_mode=None,
        backend="multiprocessing"
    ) as parallel:

        results = parallel(
            delayed(parse_and_count_entities)( data_row )
            for data_row in df_records
        )
    print('Done')

    res_df = pd.DataFrame(results)

    res_df[['Sub_Category_1', 'Sub_Category_2', 'ner_dict', 'embedding']]\
        .dropna()\
        .to_csv('prep_data.csv')

    print(res_df.head())
