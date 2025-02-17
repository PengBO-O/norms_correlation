import os
import json
import pandas as pd

EMBED_LANGUAGE = "en"
SAVE_DIR = f"./data/instructions/{EMBED_LANGUAGE}/"
WORD_RATINGS_FILE = "./data/word_ratings_en_aligned.txt"
QUERIES_FILE = "./data/Queries_v4.xlsx"
WORD_TAGS_FILE = "./data/word_ratings.txt"

def load_data():
    sheets = pd.read_excel(QUERIES_FILE, sheet_name=["noun", "verb", "adj."])
    words_tags = pd.read_csv(WORD_TAGS_FILE, encoding="utf-8", sep='\t')[["Words", "EngWords", "POS"]]
    words_attributes = pd.read_csv(WORD_RATINGS_FILE, encoding='utf-8', sep='\t')
    return sheets, words_tags, words_attributes

def generate_prompt(w, query, high_example, high_explanation, medium_example, medium_explanation):
    return (
        f'To what degree do you think of "{w}" as being associated with "{query}"? '
        f'For comparison, "{high_example}" would receive a high rating on this question because {high_explanation} '
        f'In contrast, "{medium_example}" might receive a medium rating, because {medium_explanation} '
        f'Rate the association level on a scale of 0.0 (not at all) to 6.0 (very much). '
        f'Provide only the numerical rating as your answer.\n\n'
    )

def process_attributes(words, attributes, sheets, words_tags):
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for attr in attributes:
        save_file = os.path.join(SAVE_DIR, f"{attr}.jsonl")
        with open(save_file, 'w', encoding="utf-8") as wf:
            for w in words:
                tag = words_tags[words_tags["EngWords"] == w]["POS"].iloc[0]
                query_sheet = sheets[tag]
                selected_row = query_sheet[query_sheet["Name"] == attr].iloc[0]
                
                prompt = generate_prompt(
                    w, 
                    selected_row["Query"].strip(),
                    selected_row["High Example"].strip(),
                    selected_row["High Explanation"].strip(),
                    selected_row["Medium Example"].strip(),
                    selected_row["Medium Explanation"].strip()
                )
                
                json.dump({'word': w, 'query': prompt}, wf)
                wf.write("\n")
        print(f"Complete attribute {attr}")

def main():
    sheets, words_tags, words_attributes = load_data()
    words = words_attributes["Word"].tolist()
    attributes = words_attributes.columns[2:].tolist()
    process_attributes(words, attributes, sheets, words_tags)

if __name__ == "__main__":
    main()