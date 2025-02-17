import os
import random
import re
from multiprocessing import Pool
import jsonlines
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# Global variables
NUM_PROCESSES = 64  # Adjust according to your system's capabilities
LANGUAGE = 'en'


def initialize_dataset():
    """Initialize the global dataset variable."""
    if LANGUAGE == 'zh':
        dataset = load_dataset("wikipedia", language="zh", date="20240501",
                               cache_dir="./wiki_dump_zh/", num_proc=NUM_PROCESSES, trust_remote_code=True)["train"]
        df = pd.read_csv("./data/word_ratings_zh_aligned.txt",
                         sep="\t", encoding="utf-8", header="infer")

    elif LANGUAGE == 'en':
        dataset = load_dataset("wikipedia", language="en", date="20240501",
                               cache_dir="./wiki_dump_en/", num_proc=NUM_PROCESSES, trust_remote_code=True)["train"]
        df = pd.read_csv("./data/word_ratings_en_aligned.txt",
                         sep="\t", encoding="utf-8", header="infer")
    else:
        raise ValueError(f"Language {LANGUAGE} not supported.")
    return dataset, df


def preprocess_and_select_sentences(args):
    """Preprocess text and select sentences containing the given word."""
    word, count, max_attempts, ds = args
    selected_sentences = []
    selected_indices = set()
    attempts = 0

    while count > 0 and attempts < max_attempts:
        random_index = random.randint(0, len(ds) - 1)
        if random_index in selected_indices:
            continue
        random_article = ds[random_index]
        if LANGUAGE == 'zh':
            preprocessed_text = random_article["text"].replace("\n", "。").replace("\r", "。")
            sentences = re.split(r'[。！？]', preprocessed_text)
        elif LANGUAGE == 'en':
            preprocessed_text = random_article["text"].replace("\n", ".").replace("\r", ".")
            sentences = re.split(r'[.!?]', preprocessed_text)
        else:
            raise ValueError(f"Language {LANGUAGE} not supported.")
        sentences = [sentence for sentence in sentences if word in sentence]

        if sentences:
            random_sentence = random.choice(sentences)
            save_item = {"word": word, "sentence": random_sentence}
            if save_item not in selected_sentences:
                selected_sentences.append(save_item)
                count -= 1
        attempts += 1

    return selected_sentences


def main():
    """Main function."""
    dataset, df = initialize_dataset()

    # Load word ratings
    words = df["Word"].values.tolist()

    # Configuration options
    count = 500
    max_attempts = 1000000

    # Prepare arguments for parallel processing
    args = [(word, count, max_attempts, dataset) for word in words]

    # Parallel processing with multiprocessing Pool
    with Pool(initializer=initialize_dataset, processes=NUM_PROCESSES) as pool:
        results = list(tqdm(pool.imap(preprocess_and_select_sentences, args), total=len(words)))

    # Save selected sentences to JSON Lines file
    save_path = "./selected_sentences"
    os.makedirs(save_path, exist_ok=True)
    output_file = os.path.join(save_path, f"{count}_times_{LANGUAGE}_wiki.jsonl")
    with jsonlines.open(output_file, mode="w") as writer:
        for result in results:
            for sentence in result:
                writer.write(sentence)


if __name__ == "__main__":
    main()
