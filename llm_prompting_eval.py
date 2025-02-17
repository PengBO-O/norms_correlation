import os
import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    return {'mse': mse, 'rho': rho}


def load_rating_files(directory):
    word_attr_ratings = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jsonl'):
            attribute = filename.split('.')[0]
            word_attr_ratings[attribute] = {}
            with open(os.path.join(directory, filename), 'r', encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    word_attr_ratings[attribute][item['word']] = float(item['rating'])
    return pd.DataFrame(word_attr_ratings)


def evaluation_fct(ref_df, pred_df, model, language, level):

    total_output = {}
    index_key = 'Word' if level == 'word' else None
    
    for key in (ref_df.index if index_key else ref_df.columns):
        # Extract the data
        ref_data = ref_df.loc[key] if index_key else ref_df[key]
        print(ref_data.shape)
        pred_data = pred_df.loc[key] if index_key else pred_df[key]
        print(pred_data.shape)

        # Align the DataFrames by index and columns
        aligned_ref_data, aligned_pred_data = ref_df.align(pred_df, join='outer', axis=0)

        # Compare the DataFrames element-wise
        difference_mask = aligned_ref_data != aligned_pred_data

        # Print the differences
        differences = aligned_ref_data[difference_mask]
        print("Differences in ref_data:")
        print(differences)

        differences = aligned_pred_data[difference_mask]
        print("Differences in pred_data:")
        print(differences)
        exit()
        # ref_data = ref_data.reindex(pred_data.index)

        # Drop NaN values and ensure both have the same shape
        valid_indices = ~ref_data.isna() & ~pred_data.isna()
        
        ref_data_filtered = ref_data[valid_indices]
        
        pred_data_filtered = pred_data[valid_indices]
        

        # Check if the shapes are consistent
        if ref_data_filtered.shape[0] == pred_data_filtered.shape[0]:
            eval_results = compute_metrics(
                ref_data_filtered.tolist(),
                pred_data_filtered.tolist()
            )
            total_output[key] = eval_results
        else:
            print(f"Skipping {key} due to shape mismatch after NaN removal.")
    
    output_df = pd.DataFrame(total_output).T
    avg_metrics = output_df.mean()
    std_metrics = output_df.std()
    output_df.loc['average'] = avg_metrics
    output_df.loc["std"] = std_metrics

    for col in output_df.columns:
        output_df.loc['Avg ± Std', col] = f"{avg_metrics[col]:.3f} ± {std_metrics[col]:.3f}"
    
    save_dir = f'./results/{language}'
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{model}_{level}_level.csv"
    output_df = output_df.astype(str)
    output_df.to_csv(os.path.join(save_dir, file_name), sep="\t", encoding="utf-8")
    print(f'Finished {model} on {level} level in {language} evaluation')

def main():
    for language in ['en','zh']:
        reference_file = f"./data/word_ratings_{language}_aligned.txt"
        reference_df = pd.read_csv(reference_file, encoding="utf-8", sep="\t", index_col="Word")
        reference_df = reference_df.drop('ID', axis=1)

        for model in ['llama3.1']:
            gen_rating_dir = f"./generated_outputs/{model}/{language}"
            pred_ratings_df = load_rating_files(gen_rating_dir)
            pred_ratings_df.index.name = 'Word'

            evaluation_fct(ref_df=reference_df, pred_df=pred_ratings_df, model=model, language=language, level='attribute')
            # evaluation_fct(ref_df=reference_df, pred_df=pred_ratings_df, model=model, language=language, level='word')

    print("Completed")
if __name__ == "__main__":
    main()


