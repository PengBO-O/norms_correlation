from datasets import load_dataset

dataset = load_dataset('wikipedia', language="en", date="20240501", cache_dir="./wiki_dump_en")
print(dataset['train'][0].keys())
