import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

FIGURE_SIZE = (3, 3)
FONT_SIZE = 14

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def round_down_to_nearest(value, increment):
    return np.floor(value / increment) * increment

def round_up_to_nearest(value, increment):
    return np.ceil(value / increment) * increment

def create_barplot(df, metric, model_name, language):
    save_dir = f'./figs/case_study/pos'
    os.makedirs(save_dir, exist_ok=True)
    if metric == 'rho':
        increment = 0.01
        custom_palette = {'Away': '#7895c1', 'Surprised': '#e3625d'}
    else:
        increment = 0.1
        custom_palette = {'Away': '#9bbbe1', 'Surprised': '#f09ba0'}
    # Filter the DataFrame to include only 'rho' scores
    df_metric = df[df['Metric'] == metric]
    plt.figure(figsize=(FIGURE_SIZE))
    ax = sns.barplot(x='POS', y='Value', hue='Feature', data=df_metric, errorbar=None, palette=custom_palette, width=0.3)

    min_val = round_down_to_nearest(df_metric['Value'].min(), increment)
    max_val = round_up_to_nearest(df_metric['Value'].max(), increment)
    ax.set_yticks([min_val, 0, max_val])
    if metric == 'rho':
        plt.ylabel(r'$\rho$', fontsize=FONT_SIZE, rotation=0, labelpad=0)
    else:
        plt.ylabel('MSE', fontsize=FONT_SIZE, rotation=0, labelpad=0)
    plt.legend(fontsize=10, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.xlabel('')
    # plt.ylabel('')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f'{model_name}_{language}_{metric}.eps'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, f'{model_name}_{language}_{metric}.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    

def main():
    for file in os.listdir("./results/xlingual_pos_analysis"):
        if file.endswith(".json"):
            model_name, language = file.split(".")[0].split("_")
            with open(os.path.join("./results/xlingual_analysis/pos-level", file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame.from_dict({(i, j): data[i][j] 
                             for i in data.keys() 
                             for j in data[i].keys()},
                            orient='index')
                df.reset_index(inplace=True)
                df.columns = ['Feature', 'POS', 'mse', 'rho']
                
                df_melted = df.melt(id_vars=['Feature', 'POS'], value_vars=['mse', 'rho'], 
                    var_name='Metric', value_name='Value')

                for metric in ['mse', 'rho']:
                    create_barplot(df_melted, metric, model_name, language)

                
if __name__ == "__main__":
    main()



