import transformers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from tqdm import tqdm
import copy

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model and tokenizer
model = transformers.AutoModelForSequenceClassification.from_pretrained('./refusal_classifier_small/')
tokenizer = transformers.AutoTokenizer.from_pretrained('./refusal_classifier_small/')

# Load and prepare data
def load_and_prepare_data(data_paths):
    all_data = []

    for path in data_paths:
        df = pd.read_json(path)

        # Extract language from the file path
        language = path.split('/')[-1].split('_')[0]

        # Check for required columns
        required_columns = ['input', 'answer_1', 'refusal_1', 'answer_2', 'refusal_2']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col} in file {path}")

        # Prepare positive samples (refusals)
        refusals = pd.concat([
            df[['refusal_1']].rename(columns={'refusal_1': 'text'}),
            df[['refusal_2']].rename(columns={'refusal_2': 'text'})
        ], ignore_index=True)
        refusals['label'] = 1
        refusals['language'] = language  # Add language column

        # Prepare negative samples (inputs and answers)
        non_refusals = pd.concat([
            df[['input']].rename(columns={'input': 'text'}),
            df[['answer_1']].rename(columns={'answer_1': 'text'}),
            df[['answer_2']].rename(columns={'answer_2': 'text'})
        ], ignore_index=True)
        non_refusals['label'] = 0
        non_refusals['language'] = language  # Add language column

        # Combine and shuffle
        data = pd.concat([refusals, non_refusals], ignore_index=True)

        # Ensure data is not empty before shuffling
        if data.empty:
            raise ValueError(f"Data from file {path} is empty or invalid.")

        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Handle missing values
        missing_count = data['text'].isna().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} rows with missing 'text' detected. Dropping these rows.")
        data = data.dropna(subset=['text'])

        all_data.append(data)

    # Combine data from all languages
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)  # Ensure multilingual shuffling
    return combined_data

data = load_and_prepare_data(['results/english_data.json', 'results/french_data.json', 'results/german_data.json', 'results/russian_data.json', 'results/spanish_data.json'])

# Split into train and test
_, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])

# Calculate the distribution of languages in the test set
print("Language Distribution in Test Set:")
print(test_df['language'].value_counts(normalize=True) * 100)

print("Language counts in Test Set:")
print(test_df['language'].value_counts())

# Tokenize test data
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=512)

# Convert to PyTorch Dataset
class RefusalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = RefusalDataset(test_encodings, test_df['label'].tolist())
print(f'Test dataset size: {len(test_dataset)}')

# Define DataLoader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Non-Refusal', 'Refusal'], output_dict=True)
    return {"accuracy": acc, "f1": report['Refusal']['f1-score'], "precision": report['Refusal']['precision'], "recall": report['Refusal']['recall']}

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            # Ensure logits are in the correct shape
            logits = logits.cpu().numpy()
            preds = np.argmax(logits, axis=1)
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())

    return compute_metrics(transformers.EvalPrediction(predictions=all_preds, label_ids=all_labels))

def calculate_model_size_in_millions(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

# Initial evaluation
initial_model = transformers.AutoModelForSequenceClassification.from_pretrained('./refusal_classifier_small/')
initial_model.to(device)  # Move model to device
initial_metrics = evaluate_model(initial_model, test_loader)
print(f'Initial model size: {calculate_model_size_in_millions(initial_model)}M')
print(f"Initial F1 Score: {initial_metrics['f1']}")

# Pruning loop
layers_pruned = []
current_f1 = initial_metrics['f1']

while current_f1 >= 0.8:
    min_impact_f1 = float('-inf')
    layer_to_remove = None

    # Load a new model for each pruning iteration
    pruned_model = transformers.AutoModelForSequenceClassification.from_pretrained('./refusal_classifier_small/')
    pruned_model.to(device)  # Move model to device

    # Evaluate each layer for pruning impact
    for layer_idx in tqdm(range(len(pruned_model.bert.encoder.layer) - len(layers_pruned))):
        # Create a new model for each layer evaluation
        temp_model = transformers.AutoModelForSequenceClassification.from_pretrained('./refusal_classifier_small/')
        temp_model.to(device)  # Move model to device

        for pruned_layer in layers_pruned:
            temp_model.bert.encoder.layer.pop(pruned_layer)

        # Remove the current layer
        temp_model.bert.encoder.layer.pop(layer_idx)

        # Evaluate the pruned model
        pruned_metrics = evaluate_model(temp_model, test_loader)
        pruned_f1 = pruned_metrics['f1']
        print(f'Pruned block {layer_idx}, F1: {pruned_f1}')

        # Check if this layer removal has the least impact on F1 score
        if pruned_f1 > min_impact_f1:
            min_impact_f1 = pruned_f1
            layer_to_remove = layer_idx

    if layer_to_remove is not None:
        layers_pruned.append(layer_to_remove)
        current_f1 = min_impact_f1
        print(f"Removed block {layer_to_remove}, new F1 Score: {current_f1}, new model size: {calculate_model_size_in_millions(temp_model)}M")
    else:
        break

print(f"Final blocks pruned: {layers_pruned}")
print(f"Final F1 Score: {current_f1}")
