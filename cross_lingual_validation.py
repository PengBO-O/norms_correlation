import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    return {'mse': mse, 'rho': rho}


def prediction_fct(x_train, x_test, y_train):
    regressor = Ridge(alpha=1.0).fit(x_train, y_train)
    y_pred = regressor.predict(x_test)
    return y_pred


def eval_metrics(label_df, pred_df, language, model_name, level):
    total_output = {}
    index_key = 'Word' if level == 'word' else None

    for key in (label_df.index if index_key else label_df.columns):
        eval_results = compute_metrics(
            label_df.loc[key].tolist() if index_key else label_df[key].tolist(),
            pred_df.loc[key].tolist() if index_key else pred_df[key].tolist()
        )
        total_output[key] = eval_results

    output_df = pd.DataFrame(total_output).T
    avg_metrics = output_df.mean()
    std_metrics = output_df.std()
    output_df.loc['average'] = avg_metrics
    output_df.loc["std"] = std_metrics
    output_df = output_df.astype(str)
    for col in output_df.columns:
        output_df.loc['Avg ± Std', col] = f"{avg_metrics[col]:.3f} ± {std_metrics[col]:.3f}"
    
    save_dir = f'./results/{language}'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{model_name}_{level}_level_Ridge.csv"
    output_df.to_csv(os.path.join(save_dir, file_name), sep="\t", encoding="utf-8")
    print(f'Finished {level} level evaluation')
    

WORD_DICT = "./data/word_ratings.txt"
zh_en_words = pd.read_csv(WORD_DICT, sep="\t", encoding="utf-8")[["Words", "EngWords"]]
zh2en, en2zh = dict(), dict()
for _, row in zh_en_words.iterrows():
    zh2en[row["Words"]] = row["EngWords"]
    en2zh[row["EngWords"]] = row["Words"]
language_transfer_dict = {"en": en2zh, "zh":zh2en}

embeds_dir = "./embeddings/xl_models"
embed_models = ["bert-base-multilingual-cased", "mGPT"]
train_test_languages = {"en":"zh", "zh":"en"}

for train_language, test_language in train_test_languages.items():
    train_ratings_df = pd.read_csv(f"./data/word_ratings_{train_language}_aligned.txt",encoding='utf-8', sep="\t", header='infer')
    test_ratings_df = pd.read_csv(f"./data/word_ratings_{test_language}_aligned.txt",encoding='utf-8', sep="\t", header='infer')
    attributes = train_ratings_df.columns[2:].values.tolist()

    for embed_model in embed_models:
        train_embed_file = embed_model+"_"+train_language+".pkl"
        test_embed_file = embed_model+"_"+test_language+".pkl"
        print(train_embed_file)
        print(test_embed_file)
        with open(os.path.join(embeds_dir, train_embed_file), "rb") as tr_e:
            train_embeddings = pickle.load(tr_e)
        
        with open(os.path.join(embeds_dir, test_embed_file), "rb") as te_e:
            test_embeddings = pickle.load(te_e)
        label_df_list = []
        prediction_df_list = []
        for attr in attributes:
            X_train, X_test = [],[]
            Y_train, Y_test = [],[]
            words = []
            for w, e in train_embeddings.items():
                xw = language_transfer_dict[train_language][w]
                y_train = train_ratings_df.loc[train_ratings_df["Word"]==w, attr].iloc[0]
                y_test = test_ratings_df.loc[test_ratings_df["Word"]==xw, attr].iloc[0]
                if y_train == 'na' or y_test == 'na':
                    print(f"Error word: {w}/{xw}")
                    continue
                else:
                    Y_train.append(y_train)
                    X_train.append(e.numpy())
                    if xw in test_embeddings.keys():
                        X_test.append(test_embeddings[xw].numpy())
                        Y_test.append(y_test)
                        words.append(w)
                    else:
                        continue

            X_train = np.vstack(X_train)
            X_test = np.vstack(X_test)

            Y_train = np.array(Y_train)
            Y_test = np.array(Y_test)

            Y_pred = prediction_fct(x_train=X_train, x_test=X_test, y_train=Y_train)

            pred_df = pd.DataFrame({'Word': words, attr: Y_pred})
            prediction_df_list.append(pred_df)

            label_df = pd.DataFrame({'Word': words, attr: Y_test})
            label_df_list.append(label_df)
        
        total_predictions = pd.concat(prediction_df_list).groupby('Word').first()
        total_predictions.to_csv(f"./results/all_predictions/{embed_model}_{train_language}2{test_language}.csv", sep="\t", encoding="utf-8")
        # print(total_predictions)
        # exit()
        # total_labels = pd.concat(label_df_list).groupby('Word').first()

        # eval_metrics(total_predictions, total_labels, f"{train_language}2{test_language}", embed_model, 'attribute')
        # eval_metrics(total_predictions, total_labels, f"{train_language}2{test_language}", embed_model, 'word')

                    