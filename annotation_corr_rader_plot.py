import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

# palette = sns.color_palette("colorblind", n_colors=14)

# category2color = {
#     'Vision': palette[0],  # Blue
#     'Somatic': palette[1],  # Orange
#     'Audition': palette[2],  # Green
#     'Gustation': palette[3],  # Red
#     'Olfaction': palette[4],  # Purple
#     'Motor': palette[5],  # Brown
#     'Spatial': palette[6],  # Pink
#     'Temporal': palette[7],  # Gray
#     'Causal': palette[8],  # Yellow
#     'Social': palette[9],  # Cyan
#     'Cognition': palette[10],  # Light blue
#     'Emotion': palette[11],  # Teal
#     'Drive': palette[12],  # Lavender
#     'Attention': palette[13]  # Olive
# }

n_categories = len(category2labels)
color_palette = sns.color_palette('colorblind', n_colors=n_categories)

category2color = dict(zip(category2labels.keys(), color_palette))

# category2color = {
#     'Vision': 'lightcoral',
#     'Somatic': 'slateblue',
#     'Audition': 'deepskyblue',
#     'Gustation': 'darkorange',
#     'Olfaction': 'rosybrown',
#     'Motor': 'gold',
#     'Spatial': 'black',
#     'Temporal': 'purple',
#     'Causal': 'brown',
#     'Social': 'palevioletred',
#     'Cognition': 'lime',
#     'Emotion': 'cornflowerblue',
#     'Drive': 'olivedrab',
#     'Attention': 'rosybrown'
#     }

def setup_polar_plot(ax, values, labels, color, label):
    ax.plot(X, values, label=label, color=color)
    ax.set_xticks(np.deg2rad(xticks))
    ax.set_xticklabels([]) 


    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.rad2deg(angles)

    # Add markers for each point
    ax.scatter(X[:-1], values[:-1], color=color, s=20, zorder=3)

    for label, angle in zip(labels, angles):
        ha = 'left'
        va = 'center'
        rotation = angle

        if 90 < rotation <= 270:
            rotation += 180
            ha="right"

        # label_color = next((category2color[category] for category, cat_labels in category2labels.items() if label in cat_labels), 'black')
        category = next((cat for cat, cat_labels in category2labels.items() if label in cat_labels), None)
        if category:
            label_color = category2color[category]
        else:
            label_color = 'black'

        ax.text(np.deg2rad(angle), 1.0, label, 
                rotation=rotation, ha=ha, va=va, 
                rotation_mode='anchor', 
                fontsize=18, color=label_color)

    # Adjust rticks
    rticks = np.linspace(0, 1, 5).round(2)
    ax.set_rticks(rticks, labels=[str(tick) for tick in rticks], color='grey', size=14)
    ax.set_rlim(0, 1)
    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.7, alpha=0.8)
    ax.spines['polar'].set_visible(False)

    # Manually position rticks labels to be centered
    for label in ax.get_yticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)


df = pd.read_csv('./results/zh_en_annotation_eval.csv', sep='\t',
                 index_col=0, header='infer', encoding='utf-8')
labels = df.index.values[:-3].tolist()

rho_values = list(map(float, df['rho'][:-3].values))
rho_values += rho_values[:1]

N = len(labels)
X = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
xticks = np.linspace(0, 360, N + 1)[:-1]

# Create a single figure
fig, ax = plt.subplots(figsize=(10,10),subplot_kw={'projection': 'polar'})
fig.patch.set_facecolor('white')

# Use colors from the current palette
colors = sns.color_palette("pastel")
setup_polar_plot(ax, rho_values, labels, colors[3], (1.05, 1.2))


plt.tight_layout()
plt.subplots_adjust(right=0.75)
plt.savefig('./figs/annotation_rader_plot.svg')
plt.show()
