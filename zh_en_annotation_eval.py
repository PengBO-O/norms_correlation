import pandas as pd
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict


def compute_correlation_scores(zh_scores, en_scores):
    en_cleaned = pd.to_numeric(en_scores, errors='coerce').dropna()
    zh_aligned = pd.to_numeric(zh_scores, errors='coerce').reindex(en_cleaned.index, method='ffill')
    pearson_corr, p_value_pearson = pearsonr(zh_aligned, en_cleaned)
    spearman_corr, p_value_spearman = spearmanr(zh_aligned, en_cleaned)
    mse = ((zh_aligned - en_cleaned) ** 2).mean()

    return {"r": round(pearson_corr, 3),
            "rho": round(spearman_corr, 3),
            "mse": round(mse, 3)}


zh_df = pd.read_csv("./data/word_ratings_zh_aligned.txt",
                    sep="\t", encoding="utf-8", header="infer")

en_df = pd.read_csv("./data/word_ratings_en_aligned.txt",
                    sep="\t", encoding="utf-8", header="infer")

assert zh_df.shape == en_df.shape
assert zh_df['ID'].equals(en_df['ID'])
assert zh_df.index.equals(en_df.index)

count = 0
total_r, total_rho, total_mse = 0, 0, 0
record = defaultdict(dict)
for column in zh_df.columns[2:]:
    corr_outputs = compute_correlation_scores(zh_df[column], en_df[column])
    record[column] = corr_outputs
    count += 1
    total_r += corr_outputs["r"]
    total_rho += corr_outputs["rho"]
    total_mse += corr_outputs["mse"]

mean_r = total_r / count
mean_rho = total_rho / count
mean_mse = total_mse / count

print(f"Mean R: {mean_r}")
print(f"Mean rho: {mean_rho}")
print(f"MSE: {mean_mse}")

results_df = pd.DataFrame(record).T
avg_metrics = results_df.mean()
std_metrics = results_df.std()
results_df.loc['Average'] = avg_metrics
results_df.loc['Std'] = std_metrics
avg_std_formatted = pd.DataFrame({col: f"{avg_metrics[col]:.3f} ± {std_metrics[col]:.3f}" for col in results_df.columns},index=['Avg ± Std'])
final_results_df = pd.concat([results_df, avg_std_formatted])
final_results_df.to_csv("./results/zh_en_annotation_eval.csv", sep='\t')

