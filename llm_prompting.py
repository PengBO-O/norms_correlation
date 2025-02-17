import os
import re
import json
import torch
from transformers import pipeline

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ACCESS_TOKEN = "token"

def extract_numbers(string):
    pattern = r'-?\d+(?:\.\d+)?'
    numbers = re.findall(pattern, string)
    return [float(num) if '.' in num else int(num) for num in numbers]

def initialize_pipe(model_id):
    return pipeline("text-generation", 
                    model=model_id, 
                    model_kwargs={"torch_dtype": torch.bfloat16}, 
                    device_map="auto", 
                    token=ACCESS_TOKEN)

def process_file(pipe, input_file, output_file, log_file):
    save_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": sample["query"]}
            ]
            outputs = pipe(messages, max_new_tokens=32)[0]["generated_text"][-1]
            
            try:
                ratings = extract_numbers(outputs["content"] if isinstance(outputs, dict) else outputs)
                save_items.append({"word": sample["word"], "rating": ratings[0]})
            except (IndexError, TypeError) as e:
                print(f"Failed generation on word: {sample['word']}, outputs: {outputs}")
                print(f"Error: {e}")
                with open(log_file, 'a', encoding='utf-8') as log:
                    log.write(f"'word:{sample['word']}', outputs: {outputs}")
                    log.write('\n')
                continue
    
    with open(output_file, "w", encoding="utf-8") as wf:
        for item in save_items:
            json.dump(item, wf)
            wf.write("\n")

def main():
    data_dir = "./data/instructions/en/"
    save_dir = "./generated_outputs/llama3.1"
    os.makedirs(save_dir, exist_ok=True)

    pipe = initialize_pipe(MODEL_ID)

    for file in os.listdir(data_dir):
        if file.endswith('.jsonl'):
            attr = file.split('.')[0]
            print(f"Processing {file}...")
            input_file = os.path.join(data_dir, file)
            output_file = os.path.join(save_dir, f"{attr}.jsonl")
            log_file = os.path.join(save_dir, f"{attr}_invalid.txt")
            process_file(pipe, input_file, output_file, log_file)
            print(f"{attr} completed!")

if __name__ == "__main__":
    main()
            

                

