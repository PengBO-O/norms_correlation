from numpy import save
import pandas as pd
import json
from typing import List, Tuple, Dict
from pathlib import Path


def load_values(file: Path, metric: str) -> Tuple[float, float, str, str]:
    df = pd.read_csv(file, sep='\t', header='infer')
    values = df[metric][:-3].astype(float)
    
    min_value = round(values.min(),3)
    max_value = round(values.max(),3)
    min_dim = df.loc[values.idxmin()].iloc[0]
    max_dim = df.loc[values.idxmax()].iloc[0]
    
    return min_value, max_value, str(min_dim), str(max_dim)

def get_statistics(pairs: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    outputs = []
    for model, lang_file_pair in pairs.items():
        for lang, file in lang_file_pair.items():
            file_path = Path(file)
            if not file_path.exists():
                print(f"Warning: File not found: {file}")
                continue
            
            try:
                min_value, max_value, min_dim, max_dim = load_values(file_path, 'spearman')
                outputs.append({
                    "model": model,
                    "language": lang,
                    "max_dim": max_dim,
                    "max_value": max_value,
                    "min_dim": min_dim,
                    "min_value": min_value
                })
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
    
    return outputs


def save_results(outputs: List[Dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as w:
        for item in outputs:
            w.write(json.dumps(item) + '\n')
    print(f"Results saved to {output_file}")


def min_max_statistics(pairs):
    results = get_statistics(pairs)
    save_results(results, Path("./results/statistics.jsonl"))


def mono_cross_lingual_analysis(pairs):
    results = []  # List to store results

    for model in ['mBERT', 'mGPT']:
        for (mono_lang, cross_lang) in [('ZH', 'EN2ZH'), ('EN', 'ZH2EN')]:
            mono_df = pd.read_csv(pairs[model][mono_lang], sep='\t', header='infer', index_col=0)
            mono_rhos = mono_df['spearman'][:-3].astype(float)
            cross_df = pd.read_csv(pairs[model][cross_lang], sep='\t', header='infer', index_col=0)
            cross_rhos = cross_df['spearman'][:-3].astype(float)

            gaps = mono_rhos - cross_rhos

            # Save individual gaps
            # save_df.to_csv(f"./results/{mono_lang}-{cross_lang}_{model}.csv", sep="\t", encoding="utf-8", index=False)

            # Store results
            max_index = gaps.idxmax()
            min_index = gaps.idxmin()

            results.append({
                'model': model,
                'comparison': f"{mono_lang}-{cross_lang}",
                'average_gap': gaps.mean(),
                'max_dim': gaps.index[gaps.index.get_loc(max_index)],
                'maximum': gaps.max(),
                'min_dim': gaps.index[gaps.index.get_loc(min_index)],
                'minimum': gaps.min()
            })

    # Save average gaps
    results_df = pd.DataFrame(results)
    results_df.to_csv("./results/average_gaps.csv", sep="\t", encoding="utf-8", index=False)

    print("Analysis complete. Results saved to ./results/")
    return results_df




if __name__ == "__main__":

    pairs = {
        "BERT": {
            "EN": "./results/en/bert-base-cased.csv", 
            "ZH": "./results/zh/bert-base-chinese.csv"
                },
        "GPT2": {
            "EN": "./results/en/gpt2.csv", 
            "ZH": "./results/zh/gpt2-chinese.csv"
                },
        "Llama": {
            "EN": "./results/en/llama3_evaluation.csv",
            "ZH": "./results/zh/llama3_evaluation.csv"
            },
        "QWen": {
                "EN": "./results/en/qwen2.5_evaluation.csv",
                "ZH": "./results/zh/qwen2.5_evaluation.csv"
                },
        "mBERT": {
                "EN2ZH": "./results/en2zh/bert-base-multilingual-cased.csv",
                "ZH": "./results/zh/bert-base-multilingual-cased.csv",
                "ZH2EN": "./results/zh2en/bert-base-multilingual-cased.csv",
                "EN": "./results/en/bert-base-multilingual-cased.csv"
                },
        "mGPT": {

                "EN2ZH": "./results/en2zh/mGPT.csv",
                "ZH": "./results/zh/mGPT.csv",
                "ZH2EN": "./results/zh2en/mGPT.csv",
                "EN": "./results/en/mGPT.csv"
                }
        }
    
    mono_cross_lingual_analysis(pairs)


            


            
        