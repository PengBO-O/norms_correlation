import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_SIZE = (8, 6)
FONT_SIZE = 18
BAR_WIDTH = 0.25

display_pairs = {
    "MLM":{
        "EN":{
            r"BERT$_\text{EN}$": "./results/en/bert-base-cased_en_word_level_Ridge.csv",
            r"mBERT$_\text{EN}$": "./results/en/bert-base-multilingual-cased_en_word_level_Ridge.csv",
            r"mBERT$_\text{ZH2EN}$":"./results/zh2en/bert-base-multilingual-cased_word_level_Ridge.csv"
            },
        "ZH":{
            r"BERT$_\text{ZH}$": "./results/zh/bert-base-chinese_zh_word_level_Ridge.csv",
            r"mBERT$_\text{ZH}$": "./results/zh/bert-base-multilingual-cased_zh_word_level_Ridge.csv",
            r"mBERT$_\text{EN2ZH}$": "./results/en2zh/bert-base-multilingual-cased_word_level_Ridge.csv"
            }
        },
    "CLM": {
        "EN":{
            r"GPT2$_\text{EN}$": "./results/en/gpt2_en_word_level_Ridge.csv",
            r"mGPT$_\text{EN}$": "./results/en/mGPT_en_word_level_Ridge.csv",
            r"mGPT$_\text{ZH2EN}$":"./results/zh2en/mGPT_word_level_Ridge.csv"
            },
        "ZH":{
            r"GPT2$_\text{ZH}$": "./results/zh/gpt2-chinese-cluecorpussmall_zh_word_level_Ridge.csv",
            r"mGPT$_\text{ZH}$": "./results/zh/mGPT_zh_word_level_Ridge.csv",
            r"mGPT$_\text{EN2ZH}$": "./results/en2zh/mGPT_word_level_Ridge.csv"
            }
        }
    }

raw_ratings = pd.read_csv("./data/word_ratings.txt", sep='\t', encoding='utf-8', header=0)
words_tags = raw_ratings[["Words", "EngWords", "POS"]]


def load_pred_ratings(file: str) -> pd.DataFrame:
    pred_ratings = pd.read_csv(file, sep='\t', encoding='utf-8', header=0)
    # Drop the last three rows
    pred_ratings = pred_ratings.iloc[:-3]
    return pred_ratings


def pair_words_tags(pred_ratings: pd.DataFrame, words_tags: pd.DataFrame, language: str) -> pd.DataFrame:
    # Try primary merge based on language
    if language == "EN":
        merged = pd.merge(pred_ratings, words_tags, left_on="Unnamed: 0", right_on="EngWords", how="inner")
        # If empty, try alternative merge
        if merged.empty:
            merged = pd.merge(pred_ratings, words_tags, left_on="Unnamed: 0", right_on="Words", how="inner")
    else:
        merged = pd.merge(pred_ratings, words_tags, left_on="Unnamed: 0", right_on="Words", how="inner")
        # If empty, try alternative merge
        if merged.empty:
            merged = pd.merge(pred_ratings, words_tags, left_on="Unnamed: 0", right_on="EngWords", how="inner")
    return merged


def create_pos_centric_plot(df, language, model_type, save_dir):
    # Filter data for the specified language
    lang_df = df[df['language'] == language]
    
    # Create figure
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    fig.patch.set_facecolor('white')  # White background
    
    # Get unique POS and models
    pos_types = ['adj.', 'noun', 'verb']
    models = lang_df['model'].unique()
    
    # Set x positions for bars
    x = np.arange(len(pos_types))
    
    # Use a professional color palette
    colors = sns.color_palette("pastel")[1:]  # Soft, professional colors
    
    # Plot boxes for each model
    for i, (model, color) in enumerate(zip(models, colors)):
      
        # Create boxplot for each POS
        model_data = lang_df[lang_df['model'] == model]
        pos = x + i*BAR_WIDTH
        
        # 为每个词性创建箱线图
        bp = ax.boxplot([model_data[model_data['POS'] == pos]['rho'] for pos in pos_types],
                       positions=pos,
                       widths=BAR_WIDTH*0.8,
                       patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 1.2},
                       boxprops={'facecolor': color, 'alpha': 0.7},
                       whiskerprops={'linewidth': 1.2},
                       capprops={'linewidth': 1.2},
                       flierprops={'marker': 'o', 'markerfacecolor': color, 'markersize': 5},
                       tick_labels=['' for _ in pos_types])  # 更新的参数名称
    
    # Add mean values horizontally aligned for each POS type
    y_offset = ax.get_ylim()[1] - 0.05  # Position for text
    for j, pos_type in enumerate(pos_types):
        mean_values = []
        for model in models:
            mean_val = lang_df[(lang_df['model'] == model) & 
                             (lang_df['POS'] == pos_type)]['rho'].mean()
            mean_values.append(f'{mean_val:.2f}')
        
        # Join mean values with spaces and center over the POS group
        mean_text = ' '.join(mean_values)
        ax.text(x[j] + BAR_WIDTH, y_offset, mean_text,
               ha='center', va='bottom', fontsize=FONT_SIZE-2)
    
    # Add horizontal grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.70, color='gray', zorder=0)
    
    # Customize plot
    ax.set_ylabel(r'$\rho$', fontsize=FONT_SIZE, rotation=0, labelpad=0)
    ax.set_xlabel('')
    ax.set_xticks(x + BAR_WIDTH)
    ax.set_xticklabels(pos_types, fontsize=FONT_SIZE)
    ax.tick_params(axis='y', labelsize=FONT_SIZE)
    
        # Get the actual data range
    y_min = lang_df['rho'].min()
    y_max = lang_df['rho'].max()
    
    # Add padding (e.g., 5% of the range)
    y_range = y_max - y_min
    padding = y_range * 0.05
    
    # Set y-axis limits with padding for data and text
    ax.set_ylim(y_min - padding,  # Lower limit with small padding
                y_max + y_range * 0.2)  # Upper limit with extra space for text labels
    
    # Create legend
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=model)
                      for color, model in zip(colors, models)]
    ax.legend(handles=legend_elements, fontsize=FONT_SIZE-2, frameon=False, 
             loc='upper center', bbox_to_anchor=(0.5, 1.05),
             ncol=len(models))
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Ensure grid lines are behind the boxes
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'{save_dir}/{language}_{model_type}_pos.eps', 
                format='eps',
                dpi=1200,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.savefig(f'{save_dir}/{language}_{model_type}_pos.jpg', 
                format='jpg',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()

def main():
    save_dir = "./figs/pos/"
    os.makedirs(save_dir, exist_ok=True)
    for model_type, lang_files in display_pairs.items():
        output_items = []
        for lang, model_file_pair in lang_files.items():
            for model, file in model_file_pair.items():
                # print(model)
                pred_ratings = load_pred_ratings(file)
                merged_df = pair_words_tags(pred_ratings, words_tags, lang)
                # print(merged_df.head())
                # words_per_pos = merged_df.groupby('POS').size().reset_index(name='word_count')
                # print(words_per_pos)

                # total_pos_types = merged_df['POS'].nunique()
                # print(f"Total POS types: {total_pos_types}")
                merged_df['rho'] = pd.to_numeric(merged_df['rho'], errors='coerce')
                # pos_grouped_stats = merged_df.groupby('POS')['rho'].agg(['mean', 'std']).reset_index()
                merged_df['model'] = model
                merged_df['language'] = lang
                output_items.append(merged_df)

        output_df = pd.concat(output_items, ignore_index=True)

        for lang in ['EN', 'ZH']:
            create_pos_centric_plot(output_df, lang, model_type, save_dir)


if __name__ == "__main__":
    main()
