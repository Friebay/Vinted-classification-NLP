# Vinted_classification_NLP

1. Iš daiktai.csv su nereikalingi_stulpeliai.ipynb ištriname nereikalingus stulpelius, gauname daiktai_cleaned.csv

2. Su daiktai_cleaned.csv atliekame pradinę analizę, išverčiame kategorijas ir šalis į lietuvių kalbą, gauname daiktai_translated.csv

3. Su do_lang_detect.py atliekame kalbos atpažinimą failui daiktai_translated.csv, gauname daiktai_translated_with_lang.csv

4. Su do_embed_and_ner.py atliekame embedding failui daiktai_translated_with_lang.csv, gauname daiktai_translated_with_lang_emb_ner.csv

5. Su stream_preprocess.py paruošiame failo daiktai_translated_with_lang_emb_ner.csv duomenis modelio mokymui, gauname final_df.csv