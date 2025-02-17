import pandas as pd
import numpy as np
import os
from collections import defaultdict

def load_values(file, metric: str):
    df = pd.read_csv(file, sep='\t', header='infer')
    values = df[metric].to_numpy()
    return np.append(values, values[0])  # Append the first value to the end

def compute_avg_std(values):
    return np.mean(values), np.std(values, ddof=0)

def main():
    outputs = defaultdict(lambda: defaultdict(dict))
    for root, _, files in os.walk("./results/en"):
        for file in files:
            if file.endswith(".csv"):
                model_name, method = file.split(".")[0].split("_")
                values = load_values(os.path.join(root, file), 'spearman')
                avg, std = compute_avg_std(values)

                print(f"{file}: avg = {avg:.4f}, std = {std:.4f}")
                outputs[method][model_name] = {"mean": avg, "std": std}

    for method, data in outputs.items():
        pd.DataFrame(data).T.to_csv(f"./results/en/{method.lower()}_avg_std.csv")

if __name__ == "__main__":
    main()