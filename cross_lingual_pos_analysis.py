import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    return {'mse': mse, 'rho': rho}


def pair_words_tags(pred_ratings: pd.DataFrame, words_tags: pd.DataFrame, language: str) -> pd.DataFrame:
    # Try primary merge based on language
    if language in ["en2zh", "en"]:
        merged = pd.merge(pred_ratings, words_tags, left_on="Word", right_on="EngWords", how="inner")
        # If empty, try alternative merge
        if merged.empty:
            merged = pd.merge(pred_ratings, words_tags, left_on="Word", right_on="Words", how="inner")
    elif language in ["zh2en", "zh"]:
        merged = pd.merge(pred_ratings, words_tags, left_on="Word", right_on="Words", how="inner")
        # If empty, try alternative merge
        if merged.empty:
            merged = pd.merge(pred_ratings, words_tags, left_on="Word", right_on="EngWords", how="inner")
    return merged[["Word", "Away", "Surprised", "POS"]]


# def pair_words_tags(pred_ratings: pd.DataFrame, words_tags: pd.DataFrame) -> pd.DataFrame:
#     # Try primary merge based on language

#     merged = pd.merge(pred_ratings, words_tags, left_on="Word", right_on="EngWords", how="inner")
#     # If empty, try alternative merge
#     if merged.empty:
#         merged = pd.merge(pred_ratings, words_tags, left_on="Word", right_on="Words", how="inner")

#     return merged[["Word", "Away", "Surprised", "POS"]]


file_dir = "./results/all_predictions"

WORD_DICT = "./data/word_ratings.txt"
zh_en_words = pd.read_csv(WORD_DICT, sep="\t", encoding="utf-8")[["Words", "EngWords"]]
zh2en, en2zh = dict(), dict()
for _, row in zh_en_words.iterrows():
    zh2en[row["Words"]] = row["EngWords"]
    en2zh[row["EngWords"]] = row["Words"]
language_transfer_dict = {"en2zh": en2zh, "zh2en":zh2en}

raw_ratings = pd.read_csv("./data/word_ratings.txt", sep='\t', encoding='utf-8', header=0)
words_tags = raw_ratings[["Words", "EngWords", "POS"]]

for file in os.listdir(file_dir):

    model_name,language = file.split(".")[0].split("_")
    pred_df = pd.read_csv(os.path.join(file_dir, file), encoding="utf-8", sep="\t", header="infer")
    pred_df = pair_words_tags(pred_df, words_tags, language)

    if language == "en2zh":
        label_df = pd.read_csv("./data/word_ratings_zh_aligned.txt",encoding="utf-8", sep="\t", header="infer")
    else:
        label_df = pd.read_csv("./data/word_ratings_en_aligned.txt",encoding="utf-8", sep="\t", header="infer")
    label_df = pair_words_tags(label_df, words_tags, language)

    file_output = {}
    for feature in ["Away", "Surprised"]:
        file_output[feature] = {}
        for pos_tag in pred_df["POS"].unique():
            pos_pred_df = pred_df[pred_df["POS"] == pos_tag]
            pos_label_df = label_df[label_df["POS"] == pos_tag]
            y_pred, y_label = [], []
            for w in pos_pred_df["Word"]:
                translated_w = language_transfer_dict[language][w]
                if translated_w in pos_label_df["Word"].values:
                    y_pred.append(pos_pred_df.loc[pos_pred_df["Word"] == w, feature].iloc[0])
                    y_label.append(pos_label_df.loc[pos_label_df["Word"] == translated_w, feature].iloc[0])
                else:
                    print(f"Error word: {w}/{translated_w}")
                    continue

            eval_results = compute_metrics(y_label, y_pred)
            file_output[feature][pos_tag] = eval_results

    output_dir = f"./results/xlingual_pos_analysis"
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(file_output)
    output_df.to_json(os.path.join(output_dir, f"{model_name}_{language}.json"))






        
