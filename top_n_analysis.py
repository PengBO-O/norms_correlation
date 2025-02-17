import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

save_dir = "./results/xlingual_analysis/top_n_analysis"
os.makedirs(save_dir, exist_ok=True)

WORD_DICT = "./data/word_ratings.txt"
zh_en_words = pd.read_csv(WORD_DICT, sep="\t", encoding="utf-8")[["Words", "EngWords"]]
zh2en, en2zh = dict(), dict()
for _, row in zh_en_words.iterrows():
    zh2en[row["Words"]] = row["EngWords"]
    en2zh[row["EngWords"]] = row["Words"]

language_transfer_dict = {"EN": en2zh, "ZH": zh2en}

level = "domain"

supervised_files = {
    "MLM": {
        "EN": {
            r"BERT$_\text{EN}$": f"./results/xlingual_analysis/{level}/bert_en.json",
            r"mBERT$_\text{EN}$": f"./results/xlingual_analysis/{level}/mbert_en.json",
    },
        "ZH": {
            r"BERT$_\text{ZH}$": f"./results/xlingual_analysis/{level}/bert_zh.json",
            r"mBERT$_\text{ZH}$": f"./results/xlingual_analysis/{level}/mbert_zh.json",
        }
    },
    "CLM": {
        "EN": {
            r"GPT2$_\text{EN}$": f"./results/xlingual_analysis/{level}/gpt2_en.json",
            r"mGPT$_\text{EN}$": f"./results/xlingual_analysis/{level}/mGPT_en.json",
        },
        "ZH": {
            r"GPT2$_\text{ZH}$": f"./results/xlingual_analysis/{level}/gpt2_zh.json",
            r"mGPT$_\text{ZH}$": f"./results/xlingual_analysis/{level}/mGPT_zh.json",
        }
    }
}


unsupervised_files = {
    "EN":{
        "MLM": {
            r"mBERT$_\text{ZH2EN}$": f"./results/xlingual_analysis/{level}/mbert_zh2en.json",
        },
        "CLM": {
            r"mGPT$_\text{ZH2EN}$": f"./results/xlingual_analysis/{level}/mGPT_zh2en.json",
        }
    },
    "ZH":{
        "MLM": {
            r"mBERT$_\text{EN2ZH}$": f"./results/xlingual_analysis/{level}/mbert_en2zh.json",
        },
        "CLM": {
            r"mGPT$_\text{EN2ZH}$": f"./results/xlingual_analysis/{level}/mGPT_en2zh.json",
        }
    }
}

for language, model_types in unsupervised_files.items():
    translator = language_transfer_dict[language]
    
    for model_type, unsup_models in model_types.items():
        
        for unsup_model_name, unsup_model_file in unsup_models.items():
            with open(unsup_model_file, "r") as f:
                unsup_data = json.load(f)
            unsup_model_output = {}
            for sup_model_name, sup_model_file in supervised_files[model_type][language].items():
                with open(sup_model_file, "r") as f:
                    sup_data = json.load(f)
                    
                feature_common_words = {}
                for feature, words in sup_data.items():
                    common_words_count = 0
                    for word in words:
                        translated_word = translator[word]
                        if translated_word in unsup_data[feature]:
                            # Check if ranks are equal
                            # supervised_rank = words.index(word)
                            # unsupervised_rank = unsup_data[feature].index(translated_word)
                            # if supervised_rank == unsupervised_rank:
                            common_words_count+=1
                    feature_common_words[feature] = int(common_words_count)
                
                unsup_model_output[sup_model_name] = feature_common_words
            output_df = pd.DataFrame.from_dict(unsup_model_output)
            output_df = output_df.T
            print(output_df)
            plt.figure(figsize=(10, 2))
            ax = sns.heatmap(output_df, annot=True, cmap="coolwarm")
            plt.xticks(rotation=45, fontsize=10, ha='right')
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{model_type}_{language}.png"), bbox_inches='tight')
            plt.savefig(os.path.join(save_dir, f"{model_type}_{language}.eps"), dpi=300, bbox_inches='tight')
            plt.close()


            
                    
                

