import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    return {'mse': mse, 'rho': rho}


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
    file_name = f"{model_name}_{language}_{level}_level_Ridge.csv"
    output_df.to_csv(os.path.join(save_dir, file_name), sep="\t", encoding="utf-8")
    print(f'Finished {level} level evaluation')

    
if __name__ == '__main__':

    for r, dirs, _ in os.walk("./embeddings"):
        for model_dir in dirs:
            model_language = model_dir.split('_')[0]
            for embed_r, _, embed_files in os.walk(os.path.join(r, model_dir)):
                for embed_file in embed_files:
                    model_name, embed_language = embed_file.split(".")[0].split("_")
                    with open(os.path.join(embed_r, embed_file), "rb") as rf:
                        word_embeddings = pickle.load(rf)
                    df = pd.read_csv(f"./data/word_ratings_{embed_language}_aligned.txt",
                                     encoding='utf-8', sep="\t", header='infer')
                    
                    
                    attributes = df.columns[2:].values.tolist()
                    label_df_list = []
                    prediction_df_list = []

                    for attr in attributes:
                        X, Y = [], []
                        words = []
                        for w, e in word_embeddings.items():
                            y = df.loc[df['Word'] == w, attr].iloc[0]
                            if y != 'na':

                                y = float(y)
                                X.append(e.numpy())
                                Y.append(y)
                                words.append(w)
                            else:
                                print(w, e)
                                break

                        # Convert data to numpy arrays
                        X = np.vstack(X)
                        Y = np.array(Y)

                        # Cross-validation
                        kf = KFold(n_splits=5, shuffle=True, random_state=42)
                        # model = LinearRegression()
                        model = Ridge(alpha=1.0)
                        predictions = cross_val_predict(model, X, Y, cv=kf)

                        pred_df = pd.DataFrame({'Word': words, attr: predictions})
                        prediction_df_list.append(pred_df)

                        label_df = pd.DataFrame({'Word': words, attr: Y})
                        label_df_list.append(label_df)

                    # Display results
                    total_predictions = pd.concat(prediction_df_list).groupby('Word').first()
            
                    total_predictions.to_csv(f"./results/all_predictions/{model_name}_{embed_language}.csv", sep="\t", encoding="utf-8")
                    # total_labels = pd.concat(label_df_list).groupby('Word').first()

                    # eval_metrics(total_predictions, total_labels, embed_language, model_name, 'attribute')
                    # eval_metrics(total_predictions, total_labels, embed_language, model_name, 'word')
