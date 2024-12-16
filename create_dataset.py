from datasets import Dataset, DatasetDict
import json
import os

# Define the directory containing your JSON files
directory = "results/"

# Function to load a JSON file into a Hugging Face Dataset
def load_json_as_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Since the data is a list of dictionaries, use Dataset.from_list
    return Dataset.from_list(data)

# Load all JSON files and create a DatasetDict
dataset_dict = {}

for file_name in os.listdir(directory):
    if file_name.endswith(".json"):
        # Extract the language code (e.g., 'english', 'french')
        lang_code = file_name.split('_')[0]
        
        # Load the data
        file_path = os.path.join(directory, file_name)
        dataset_dict[lang_code] = load_json_as_dataset(file_path)

# Convert the dictionary into a DatasetDict
dataset = DatasetDict(dataset_dict)

# Optionally, you can print or inspect the dataset
print(dataset)

# Save the dataset to disk (optional)
dataset.save_to_disk("multilingual_refusals")

