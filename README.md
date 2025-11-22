# Vinted_classification_NLP

## Failai

1. Iš daiktai.csv su `nereikalingi_stulpeliai.ipynb` ištriname nereikalingus stulpelius, gauname `daiktai_cleaned.csv`.

2. Su `daiktai_cleaned.csv` atliekame pradinę analizę, išverčiame kategorijas ir šalis į lietuvių kalbą, gauname `daiktai_translated.csv`.

3. Paleidę `prepare_nlp.py` gauname `final_df.csv`, kurį naudojame modelių mokymui ir testavimui.

## Kodo šaltiniai

Šio darbo metu kai kurias kodo dalis rašydami naudojomės dokumentacija, forumais ir dirbtiniu intelektu:

- Faile `prepare_nlp.py` taisant `joblib` naudota perplexity pagalba: https://www.perplexity.ai/search/why-may-joblib-return-error-bu-gsvyExekQkOLnsazBWsVwg#0
- Rašant `joblib` kodą remtasi paketo dokumentacija.
- Ištrynimas html ir el. laiškų: https://www.perplexity.ai/search/does-spacy-strip-emails-html-b-uJXFh9i2QZ2xZonMy6IuNw#1
Braižant grafikus buvo remtasi forumo klausimais:
- https://stackoverflow.com/questions/42128467/plot-multiple-columns-of-pandas-dataframe-on-the-bar-chart
- https://stackoverflow.com/questions/54506626/how-to-understand-seaborns-heatmap-annotation-format
- https://stackoverflow.com/questions/32428193/saving-matplotlib-graphs-to-image-as-full-screen
- Pasitkslinimui kaip logistinėje regresijoje scikit-learn naudojami `class_weights`: https://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
- `joblib` naudojimo idėją paėmėme iš: https://stackoverflow.com/questions/54201004/multithreading-with-spacy-is-joblib-necessary
Naudoti būtent šie dokumentacijos puslapiai:
- https://scikit-learn.org/0.24/modules/generated/sklearn.metrics.plot_confusion_matrix.html
- https://scikit-learn.org/0.24/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ParameterSampler.html#sklearn.model_selection.ParameterSampler
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- https://scikit-learn.org/stable/modules/ensemble.html
