import os
from typing import List, Tuple, Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from regex import D
import seaborn as sns

# Use Seaborn to set the color palette
# sns.set_palette()  # or "pastel" for softer colors
# sns.color_palette("hls", 8)

category2labels = {
    'Vision': ['Vision', 'Bright', 'Dark', 'Color', 'Pattern', 'Large', 'Small', 'Motion', 'Biomotion', 'Fast', 'Slow', 'Shape', 'Complexity', 'Face', 'Body'],
    'Attention': ['Attention', 'Arousal'],
    'Somatic': ['Touch', 'Temperature', 'Texture', 'Weight', 'Pain', "Hot", "Cold", "Smooth", "Rough", "Light", "Heavy"],
    'Cognition': ['Cognition'],
    'Audition': ['Audition', 'Loud', 'Low', 'High', 'Sound', 'Music', 'Speech'],
    'Gustation': ['Taste'],
    'Motor': ['Head', 'UpperLimb', 'LowerLimb', 'Practice'],
    'Olfaction': ['Smell'],
    'Spatial': ['Landmark', 'Path', 'Scene', 'Near', 'Toward', 'Away', 'Number'],
    'Temporal': ['Time', 'Duration', 'Long', 'Short'],
    'Causal': ['Caused', 'Consequential'],
    'Social': ['Social', 'Human', 'Communication', 'Self'],
    'Emotion': ['Benefit', 'Harm', 'Pleasant', 'Unpleasant', 'Happy', 'Sad', 'Angry', 'Disgusted', 'Fearful', 'Surprised'],
    'Drive': ['Drive', 'Needs'],
    }


# Load data
monolingual_display_pairs = {
    "BERT": {
        "EN": "./results/en/bert-base-cased.csv", 
        "ZH": "./results/zh/bert-base-chinese.csv"
            },
    "GPT2": {
        "EN": "./results/en/gpt2.csv", 
        "ZH": "./results/zh/gpt2-chinese.csv"
            },
    "mBERT": {
            "EN": "./results/en/bert-base-multilingual-cased.csv",
            "ZH": "./results/zh/bert-base-multilingual-cased.csv",
            },
    "mGPT": {
            "EN": "./results/en/mGPT.csv", 
            "ZH": "./results/zh/mGPT.csv",
            },
    "Llama": {
        "EN": "./results/en/llama3_evaluation.csv",
        "ZH": "./results/zh/llama3_evaluation.csv"
        },
        "QWen": {
            "EN": "./results/en/qwen2.5_evaluation.csv",
            "ZH": "./results/zh/qwen2.5_evaluation.csv"
            }
    }

crosslingual_display_pairs = {    
    "mBERT": {
            "EN2ZH": "./results/en2zh/bert-base-multilingual-cased.csv",
            "ZH2EN": "./results/zh2en/bert-base-multilingual-cased.csv"
            },
    "mGPT": {

            "EN2ZH": "./results/en2zh/mGPT.csv",
            "ZH2EN": "./results/zh2en/mGPT.csv"
            }
    }

mixed_display_pairs = {    
    "mBERT": {
            "EN": "./results/en/bert-base-multilingual-cased.csv",
            "ZH2EN": "./results/zh2en/bert-base-multilingual-cased.csv",
            "ZH": "./results/zh/bert-base-multilingual-cased.csv",
            "EN2ZH": "./results/en2zh/bert-base-multilingual-cased.csv"
            },
    "mGPT": {
            "EN": "./results/en/mGPT.csv", 
            "ZH2EN": "./results/zh2en/mGPT.csv",
            "ZH": "./results/zh/mGPT.csv",
            "EN2ZH": "./results/en2zh/mGPT.csv"
            }
    }

mlm_display_pairs = {
    "EN":{
        r"BERT$_\text{EN}$": "./results/en/bert-base-cased.csv",
        r"mBERT$_\text{EN}$": "./results/en/bert-base-multilingual-cased.csv",
        r"mBERT$_\text{ZH2EN}$":"./results/zh2en/bert-base-multilingual-cased.csv"
    },
    "ZH":{
        r"BERT$_\text{ZH}$": "./results/zh/bert-base-chinese.csv",
        r"mBERT$_\text{ZH}$": "./results/zh/bert-base-multilingual-cased.csv",
        r"mBERT$_\text{EN2ZH}$": "./results/en2zh/bert-base-multilingual-cased.csv"
    }
}


clm_display_pairs = {
    "EN":{
        r"GPT2$_\text{EN}$": "./results/en/gpt2.csv",
        r"mGPT$_\text{EN}$": "./results/en/mGPT.csv",
        r"mGPT$_\text{ZH2EN}$":"./results/zh2en/mGPT.csv"
    },
    "ZH":{
        r"GPT2$_\text{ZH}$": "./results/zh/gpt2-chinese.csv",
        r"mGPT$_\text{ZH}$": "./results/zh/mGPT.csv",
        r"mGPT$_\text{EN2ZH}$": "./results/en2zh/mGPT.csv"
    }
}

FIGURE_SIZE = (10, 10)
FONT_SIZE = 18
MARKER_SIZE = 14
GRID_ALPHA = 0.6


n_categories = len(category2labels)
color_palette = sns.color_palette('colorblind', n_colors=n_categories)

category2color = dict(zip(category2labels.keys(), color_palette))

def round_down_to_nearest(value, increment):
    return np.floor(value / increment) * increment

def round_up_to_nearest(value, increment):
    return np.ceil(value / increment) * increment

def load_values(file: str, metric: str) -> Tuple[List[float], List[str]]:
    df = pd.read_csv(file, sep='\t', header='infer')
    values = df[metric][:-3].astype(float).tolist()
    values.append(values[0])  # Append the first value to the end
    labels = df.iloc[:, 0].tolist()[:-3]
    return values, labels

def load_df(file: str) -> pd.DataFrame:
    df = pd.read_csv(file, sep='\t', header='infer', index_col=0)
    return df

def get_domain_mean(df: pd.DataFrame, domain_features: dict) -> pd.DataFrame:
    domain_mean_rho = {}
    for domain, features in domain_features.items():
        domain_rhos = [float(df['spearman'][feature]) for feature in features if feature in df.index]
        if domain_rhos:  # Check if domain_rhos is not empty
            domain_mean_rho[domain] = {'rho': np.mean(domain_rhos), 'std': np.std(domain_rhos)}
        else:
            domain_mean_rho[domain] = {'rho': np.nan, 'std': np.nan}  # Handle empty case
    return pd.DataFrame(domain_mean_rho).T


def setup_feature_polar_plot(ax: plt.Axes, values: List[float], labels: List[str], color: str, line_label: str, ax_size=1.0) -> None:
    N = len(labels)
    X = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    xticks = np.linspace(0, 360, N + 1)[:-1]

    line_style = '-' if any(sub in line_label for sub in ['EN2ZH', 'ZH2EN']) else '--'

    ax.plot(X, values, label=line_label, color=color, linestyle=line_style, marker='o' if any(sub in line_label for sub in ['EN2ZH', 'ZH2EN']) else 'x' )
    ax.set_xticks(np.deg2rad(xticks))
    ax.set_xticklabels([])

    angles = np.deg2rad(np.linspace(0, 360, len(labels), endpoint=False))

    ax.scatter(X[:-1], values[:-1], color=color, s=MARKER_SIZE, zorder=3)

    for label, angle in zip(labels, angles):
        ha = 'right' if 90 < np.rad2deg(angle) <= 270 else 'left'
        rotation = np.rad2deg(angle) + 180 if 90 < np.rad2deg(angle) <= 270 else np.rad2deg(angle)
        
        label_color = next((category2color[category] for category, cat_labels in category2labels.items() if label in cat_labels), 'black')
        
        ax.text(angle, ax_size, label, 
                rotation=rotation, ha=ha, va='center', 
                rotation_mode='anchor', fontsize=FONT_SIZE,
                color=label_color)

    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.7, alpha=GRID_ALPHA)
    ax.spines['polar'].set_visible(False)

    for label in ax.get_yticklabels():
        label.set_ha('center')
        label.set_va('center')

    ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.22), fontsize=FONT_SIZE-2)



def setup_domain_polar_plot(ax: plt.Axes, df: pd.DataFrame, color: str, line_label: str, ax_size=1.0) -> None:
    labels = df.index.tolist()
    N = len(labels)
    X = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    xticks = np.linspace(0, 360, N + 1)[:-1]

    line_style = '-' if any(sub in line_label for sub in ['EN2ZH', 'ZH2EN']) else '--'

    mean_rhos = df['rho'].values
    std_rhos = df['std'].values

    # Handle NaN or Inf values
    mean_rhos = np.nan_to_num(mean_rhos, nan=0.0, posinf=0.0, neginf=0.0)
    std_rhos = np.nan_to_num(std_rhos, nan=0.0, posinf=0.0, neginf=0.0)

    # Adjust rticks
    # min_value = np.min(mean_rhos)
    # max_value = np.max(mean_rhos)

    # Round min and max values to the nearest increment
    # increment = 0.1
    # min_value_rounded = round_down_to_nearest(min_value, increment)
    # max_value_rounded = round_up_to_nearest(max_value, increment)

    mean_rhos = np.append(mean_rhos, mean_rhos[0])
    std_rhos = np.append(std_rhos, std_rhos[0])

    # color = sns.color_palette("pastel")[3]

    ax.plot(X, mean_rhos, label=line_label, color=color, linestyle=line_style, marker='o' if any(sub in line_label for sub in ['EN2ZH', 'ZH2EN']) else 'x' )
    ax.errorbar(X[:-1], mean_rhos[:-1], yerr=std_rhos[:-1], fmt='o', color=color, 
                ecolor=color, elinewidth=2, capsize=4)
    
    ax.set_xticks(np.deg2rad(xticks))
    ax.set_xticklabels([]) 

    angles = np.deg2rad(np.linspace(0, 360, len(labels), endpoint=False))

    # Add markers for each point
    ax.scatter(X[:-1], mean_rhos[:-1], color=color, s=20, zorder=3)

    for label, angle in zip(labels, angles):
        ha = 'right' if 90 < np.rad2deg(angle) <= 270 else 'left'
        rotation = np.rad2deg(angle) + 180 if 90 < np.rad2deg(angle) <= 270 else np.rad2deg(angle)

        label_color = category2color[label]
        ax.text(angle, ax_size, label, 
                rotation=rotation, ha=ha, va='center', 
                rotation_mode='anchor', 
                fontsize=FONT_SIZE,
                color=label_color)

    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.spines['polar'].set_visible(False)

    for label in ax.get_yticklabels():
        label.set_ha('center')
        label.set_va('center')

    ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.22), fontsize=FONT_SIZE-2)


def plot_language_pairs(display_pairs: Dict[str, Dict[str, str]], save_dir: str, is_monolingual: bool = True) -> None:
    os.makedirs(save_dir, exist_ok=True)
    # rocket_colors = sns.color_palette('husl')
    # mono_colors = [rocket_colors[4], rocket_colors[0]]

    # set2_colors = sns.color_palette("Set2")
    # cross_colors = set2_colors[:2]
    line_palette = sns.color_palette("pastel", n_colors=4) # Adjust n_colors based on the maximum number of lines in any plot

    for model, lang_file_pair in display_pairs.items():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('white')

        rticks = np.linspace(0.0, 1, 5).round(2) if is_monolingual else [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_rticks(rticks, labels=[str(tick) for tick in rticks], color='slategray', size=FONT_SIZE-2)
        ax.set_rlim(0.0, 1) if is_monolingual else ax.set_rlim(-0.2, 1)
        if is_monolingual:
            line_colors = line_palette[:2]
        else:
            line_colors = line_palette[2:]

        for i, (lang, file) in enumerate(lang_file_pair.items()):
            values, labels = load_values(file, "spearman")
            setup_feature_polar_plot(ax, values, labels, line_colors[i], lang)

        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        if is_monolingual:
            plt.savefig(os.path.join(save_dir, f'{model}_mono.jpg'), bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{model}_mono.eps'), dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(save_dir, f'{model}_cross.jpg'), bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f'{model}_cross.eps'), dpi=300, bbox_inches='tight')

        plt.close(fig)


def mixed_plot(display_pairs: Dict[str, Dict[str, str]], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    line_palette = sns.color_palette("Paired")[2:6]

    for model, lang_file_pair in display_pairs.items():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('white')

        rticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8]
        ax.set_rticks(rticks, labels=[str(tick) for tick in rticks], color='slategray', size=FONT_SIZE)
        ax.set_rlim(-0.2, 0.8)

        for i, (lang, file) in enumerate(lang_file_pair.items()):
            values, labels = load_values(file, "spearman")
            setup_feature_polar_plot(ax, values, labels, line_palette[i], lang)
        
        plt.tight_layout()
        # plt.subplots_adjust(left=0.2)
        plt.savefig(os.path.join(save_dir, f'{model}.jpg'), bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{model}.eps'), dpi=300, bbox_inches='tight')
        plt.close(fig)


def family_plot(display_pairs: Dict[str, Dict[str, str]], save_dir: str, is_mlm: bool = True) -> None:
    os.makedirs(save_dir, exist_ok=True)
    line_colors = sns.color_palette("pastel")[1:]
    for lang, model_file_pair in display_pairs.items():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('white')

        # Initialize min and max values
        min_value, max_value = float('inf'), float('-inf')
        
        # Calculate min and max values across all models
        for model, file in model_file_pair.items():
            values, _ = load_values(file, "spearman")
            min_value = min(min_value, min(values))
            max_value = max(max_value, max(values))
        
        # Round min and max values to the nearest increment
        increment = 0.1
        min_value_rounded = round_down_to_nearest(min_value, increment)
        max_value_rounded = round_up_to_nearest(max_value, increment)
        
        # Set rticks and rlim based on min and max values
        rticks = np.linspace(min_value_rounded, max_value_rounded, num=5).round(2)
        ax.set_rticks(rticks, labels=[str(tick) for tick in rticks], color='black', size=FONT_SIZE)
        ax.set_rlim(min_value_rounded, max_value_rounded)


        for i, (model, file) in enumerate(model_file_pair.items()):
            values, labels = load_values(file, "spearman")
            setup_feature_polar_plot(ax, values, labels, line_colors[i], line_label=model, ax_size=max_value_rounded)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{lang}_{"mlm" if is_mlm else "clm"}.jpg'), bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{lang}_{"mlm" if is_mlm else "clm"}.eps'), dpi=300, bbox_inches='tight')


def family_domain_plot(display_pairs: Dict[str, Dict[str, str]], save_dir: str, is_mlm: bool = True) -> None:
    os.makedirs(save_dir, exist_ok=True)
    line_colors = sns.color_palette("pastel")[1:]
    for lang, model_file_pair in display_pairs.items():
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('white')

        min_value, max_value = float('inf'), float('-inf')
        
        # Calculate min and max values across all models
        for model, file in model_file_pair.items():
            values, _ = load_values(file, "spearman")
            min_value = min(min_value, min(values))
            max_value = max(max_value, max(values))

        increment = 0.1
        min_value_rounded = round_down_to_nearest(min_value, increment)
        max_value_rounded = round_up_to_nearest(max_value, increment)

        rticks = np.linspace(min_value_rounded, max_value_rounded, num=5).round(2)
        ax.set_rticks(rticks, labels=[str(tick) for tick in rticks], color='black', size=FONT_SIZE)
        ax.set_rlim(min_value_rounded, max_value_rounded)

        for i, (model, file) in enumerate(model_file_pair.items()):
            df = load_df(file)
            domain_mean_rho = get_domain_mean(df, category2labels)
            setup_domain_polar_plot(ax, domain_mean_rho, line_colors[i], line_label=model, ax_size=max_value_rounded)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{lang}_{"mlm" if is_mlm else "clm"}.jpg'), bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'{lang}_{"mlm" if is_mlm else "clm"}.eps'), dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    # mono_save_dir = "./figs/"
    # cross_save_dir = "./figs/"
    # mixed_save_dir = "./figs/mixed/"
    # plot_language_pairs(monolingual_display_pairs, mono_save_dir)
    # plot_language_pairs(crosslingual_display_pairs, cross_save_dir, is_monolingual=False)
    # mixed_plot(mixed_display_pairs, mixed_save_dir)
    # family_save_dir = "./figs/family/"
    # family_plot(mlm_display_pairs, family_save_dir, is_mlm=True)
    # family_plot(clm_display_pairs, family_save_dir, is_mlm=False)

    family_domain_plot(mlm_display_pairs, "./figs/family_domain/", is_mlm=True)
    family_domain_plot(clm_display_pairs, "./figs/family_domain/", is_mlm=False)


