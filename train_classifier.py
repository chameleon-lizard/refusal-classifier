import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import (
    XLMRobertaTokenizerFast,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Train XLM-R classifier to detect refusals.")
    parser.add_argument("--data_paths", type=str, nargs='+', required=True, 
                        help="Paths to the JSON datasets for different languages.")
    parser.add_argument("--model_size", type=str, default="xlm-roberta-base", 
                        choices=["xlm-roberta-base", "xlm-roberta-large"],
                        help="XLM-R model size to use.")
    parser.add_argument("--output_dir", type=str, default="./refusal_classifier", 
                        help="Directory to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    return args

def load_and_prepare_data(data_paths):
    all_data = []

    for path in data_paths:
        df = pd.read_json(path)

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

        # Prepare negative samples (inputs and answers)
        non_refusals = pd.concat([
            df[['input']].rename(columns={'input': 'text'}),
            df[['answer_1']].rename(columns={'answer_1': 'text'}),
            df[['answer_2']].rename(columns={'answer_2': 'text'})
        ], ignore_index=True)
        non_refusals['label'] = 0

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

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=512)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=['Non-Refusal', 'Refusal'], output_dict=True)
    return {"accuracy": acc, "f1": report['Refusal']['f1-score'], "precision": report['Refusal']['precision'], "recall": report['Refusal']['recall']}

def main():
    args = parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and prepare data
    data = load_and_prepare_data(args.data_paths)

    # Split into train and test
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=args.seed, stratify=data['label'])

    # Load tokenizer
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(args.model_size)

    # Tokenize data
    train_encodings = tokenize_function(train_df.to_dict(orient='list'), tokenizer)
    test_encodings = tokenize_function(test_df.to_dict(orient='list'), tokenizer)

    # Convert to torch Dataset
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

    train_dataset = RefusalDataset(train_encodings, train_df['label'].tolist())
    test_dataset = RefusalDataset(test_encodings, test_df['label'].tolist())

    # Load model
    model = XLMRobertaForSequenceClassification.from_pretrained(args.model_size, num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=args.seed,
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value}")

    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

