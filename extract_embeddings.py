import os
import argparse
import pickle
from collections import defaultdict
import jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, MT5Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and save word embeddings for given sentences.")
    parser.add_argument('--model_language', type=str, default='xl',
                        choices=['zh', 'en', 'xl'], help="Language of the model to use")
    return parser.parse_args()


def get_word_embeddings(word_sentences, tokenizer, language, model, device):
    least_frequent_word = min(word_sentences, key=lambda k: len(word_sentences[k]))
    word_embeddings = {}

    for word, sentences in tqdm(word_sentences.items(), desc="Processing words"):
        embeddings = []

        for sentence in sentences:
            if type(tokenizer).__name__ == 'GPT2TokenizerFast':
                if word == sentence.split()[0]:
                    word_tokens = tokenizer.tokenize(word)
                else:
                    word_tokens = tokenizer.tokenize(' ' + word)
            elif type(tokenizer).__name__ == 'T5Tokenizer' and language == 'zh':
                word_tokens = tokenizer.tokenize(word)
                if len(word_tokens) > 1:
                    word_tokens = word_tokens[1:]

            else:
                word_tokens = tokenizer.tokenize(word)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            tokenized_inputs = tokenizer(sentence,
                                         return_tensors='pt',
                                         truncation=True,
                                         max_length=512,
                                         add_special_tokens=True).to(device)

            with torch.no_grad():
                outputs = model(**tokenized_inputs)

            input_ids = tokenized_inputs['input_ids'][0]
            word_indices = [i for i, token in enumerate(input_ids) if token in word_token_ids]

            if word_indices:
                token_hidden_states = outputs.last_hidden_state[0, word_indices]
                embeddings.append(token_hidden_states.mean(dim=0) if len(word_indices) > 1 else token_hidden_states[0])

        if embeddings:
            word_embeddings[word] = torch.stack(embeddings).mean(dim=0).cpu()
            print(f"Finished processing '{word}'")
        else:
            print(f"No embeddings found for '{word}'")

    print(f"Word '{least_frequent_word}' has the fewest sentences: {len(word_sentences[least_frequent_word])}")
    return word_embeddings


def check_embedding_shapes(word_embeddings, expected_shape=768):
    valid = True
    for word, embedding in word_embeddings.items():
        if embedding.shape != (expected_shape,):
            print(f"Shape mismatch for '{word}': Expected {expected_shape}, got {embedding.shape}")
            valid = False
    if valid:
        print("All embeddings have consistent shapes.")
    return valid


def save_word_embeddings(word_embeddings, save_path, language, model_name):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{model_name}_{language}.pkl")
    with open(file_path, 'wb') as wf:
        pickle.dump(word_embeddings, wf)
    print(f"Embeddings saved to '{file_path}'")


def process_language_embeddings(model_id, word_sentences, save_path, language, device):
    if model_id == "uer/gpt2-chinese-cluecorpussmall":
        tokenizer = BertTokenizer.from_pretrained(model_id)
    elif model_id == "THUMT/mGPT":
        tokenizer = MT5Tokenizer.from_pretrained(model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    word_embeddings = get_word_embeddings(word_sentences, tokenizer, language, model, device)

    if check_embedding_shapes(word_embeddings, model.config.hidden_size):
        save_word_embeddings(word_embeddings, save_path, language, model_id.split("/")[-1])
    else:
        print("Error: Inconsistent embedding shapes")


def load_word_sentences(file_path):
    word_sentences = defaultdict(list)
    with jsonlines.open(file_path, 'r') as rf:
        for obj in rf:
            word_sentences[obj['word']].append(obj['sentence'])
    return word_sentences


def main():
    args = parse_args()
    #
    model_dict = {
        "en": ["google-bert/bert-base-cased", "openai-community/gpt2"],
        "zh": ["google-bert/bert-base-chinese", "uer/gpt2-chinese-cluecorpussmall"],
        # "xl": ["google-bert/bert-base-multilingual-cased", "THUMT/mGPT"]
        "xl": ["google-bert/bert-base-multilingual-cased"]
    }

    model_ids = model_dict[args.model_language]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = f"./embeddings/{args.model_language}_models"

    if args.model_language == "xl":
        for tgt_language in ['en', 'zh']:
            file_path = f'./selected_sentences/500_{tgt_language}_wiki.jsonl'
            word_sentences = load_word_sentences(file_path)
            for model_id in model_ids:
                process_language_embeddings(model_id, word_sentences, save_path, tgt_language, device)
    else:
        file_path = f'./selected_sentences/500_{args.model_language}_wiki.jsonl'
        word_sentences = load_word_sentences(file_path)
        for model_id in model_ids:
            process_language_embeddings(model_id, word_sentences, save_path, args.model_language, device)


if __name__ == "__main__":
    main()
