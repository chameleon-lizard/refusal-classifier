import os
import tarfile
import urllib.request
import urllib.error
import transformers
from collections import Counter
import itertools

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BertModel, RobertaModel

def msize(model):
    """Calculates the number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())

def embed(sentences, model, tokenizer, device):
    """
    Generates sentence embeddings.
    """
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)

        if isinstance(model, BertModel) or isinstance(model, transformers.BertForSequenceClassification):
            embeddings = model_output.hidden_states[-1][:, 0, :]
        elif isinstance(model, RobertaModel) or isinstance(model, transformers.RobertaForSequenceClassification):
            embeddings = model_output.hidden_states[-1][:, 0, :]
        else:
            raise ValueError("Unsupported model type: must be Bert or Roberta")

        embeddings = torch.nn.functional.normalize(embeddings)

    return embeddings.cpu().numpy()

def download_and_extract_corpora(corpus_urls, data_dir="data"):
    """
    Downloads and extracts corpora from provided URLs.
    """
    corpus_dirs = {}
    os.makedirs(data_dir, exist_ok=True)
    for url in tqdm(corpus_urls, desc="Processing corpus URLs"):
        filename = os.path.basename(url)
        language_code = filename.split("-")[0]
        corpus_file = os.path.join(data_dir, f"{language_code}.tar.gz")
        corpus_dir = os.path.join(data_dir, filename.replace(".tar.gz", ""))
        if not os.path.exists(corpus_file):
            print(f"\nDownloading corpus for '{language_code}' from {url}...")
            try:
                urllib.request.urlretrieve(url, corpus_file)
                print(f"Downloaded '{corpus_file}'.")
            except urllib.error.URLError as e:
                print(f"Failed to download {url}. Error: {e}")
                continue
        else:
            print(f"\nCorpus file '{corpus_file}' already exists. Skipping download.")
        if not os.path.exists(corpus_dir):
            print(f"Extracting '{corpus_file}'...")
            try:
                with tarfile.open(corpus_file, "r:gz") as tar:
                    tar.extractall(path=data_dir)
                print(f"Extracted to '{corpus_dir}'.")
            except tarfile.TarError as e:
                print(f"Failed to extract {corpus_file}. Error: {e}")
                continue
        else:
            print(f"Corpus directory '{corpus_dir}' already exists. Skipping extraction.")
        corpus_dirs[language_code] = corpus_dir
    return corpus_dirs

def find_sentences_files(corpus_dir):
    """
    Finds all files containing sentences in the given corpus directory.
    """
    sentence_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file_name in files:
            if any(keyword in file_name.lower() for keyword in ["sentences", "text", "sent"]):
                file_path = os.path.join(root, file_name)
                print(f"Found sentences file: {file_path}")
                sentence_files.append(file_path)
    if not sentence_files:
        print(f"No sentences files found in '{corpus_dir}'.")
    return sentence_files

def read_sentences_with_fallback(file_paths):
    """
    Reads sentences from multiple files, handling potential encoding issues.
    """
    all_sentences = []
    encodings = ['utf-8', 'windows-1251', 'latin-1']
    for file_path in file_paths:
        sentences = []
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    sentences.extend([line.strip() for line in f])
                    print(f"Successfully read {file_path} using {encoding} encoding.")
                    break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {file_path} with encoding {encoding}: {e}. Skipping.")
                continue
        else:
            print(f"Failed to read {file_path} with all tried encodings.")
        all_sentences.extend(sentences)
    if not all_sentences:
        return None
    return all_sentences

def create_new_embedding(old_model, new_vocab_size, hidden_size, selected_vocab, device):
    """
    Creates a new embedding layer, copying weights from the old model.
    Handles both Bert and Roberta models.
    """
    if isinstance(old_model, BertModel) or isinstance(old_model, transformers.BertForSequenceClassification):
        new_embedding = torch.nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,  # Assuming BERT-like model
        ).to(device)
        new_embedding.weight.data.copy_(old_model.bert.embeddings.word_embeddings.weight[selected_vocab, :].to(device))
    elif isinstance(old_model, RobertaModel) or isinstance(old_model, transformers.RobertaForSequenceClassification):
        new_embedding = torch.nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=hidden_size,
            padding_idx=1,  # Assuming RoBERTa-like model
        ).to(device)
        new_embedding.weight.data.copy_(old_model.roberta.embeddings.word_embeddings.weight[selected_vocab, :].to(device))
    else:
        raise ValueError("Unsupported model type: must be Bert or Roberta")
    return new_embedding


def find_n_letter_tokens(tokenizer, alphabets, max_n=3):
    """
    Finds tokens in the tokenizer's vocabulary that correspond to n-letter combinations
    from the given alphabets.

    This version iterates over the tokenizer's vocabulary for better efficiency.
    It also combines n-letter tokens from all languages into one set for faster lookups.

    Args:
        tokenizer: The original tokenizer.
        alphabets (dict): A dictionary where keys are language codes (e.g., "en", "ru") and
                          values are strings representing the alphabet of that language.
        max_n (int): The maximum number of letters to consider for combinations (e.g., 3 for
                     single, double, and triple-letter combinations).

    Returns:
        set: A set of token IDs corresponding to the n-letter combinations found in the
             tokenizer's vocabulary.
    """

    n_letter_tokens = set()
    all_combinations = set()  # Combine combinations across languages

    # Generate all n-letter combinations in one set
    for n in range(1, max_n + 1):
        print(f"Generating {n}-letter combinations...")
        for lang, alphabet in alphabets.items():
            print(f"  Processing language: {lang}")
            for combination in itertools.product(alphabet, repeat=n):
                 all_combinations.add("".join(combination))


    print(f"Checking vocabulary for n-letter tokens...")
    for token_str, token_id in tqdm(tokenizer.vocab.items(), desc="Checking tokens"):
        token_to_check = token_str if not token_str.startswith('▁') else token_str[1:]
        if token_to_check in all_combinations:
            n_letter_tokens.add(token_id)

    return n_letter_tokens

def strip_model(
    model_name,
    new_model_name,
    vocab_threshold=5,
    corpus_urls = [
        "https://downloads.wortschatz-leipzig.de/corpora/rus-ru_web-public_2019_1M.tar.gz",
        "https://downloads.wortschatz-leipzig.de/corpora/eng-com_web-public_2018_1M.tar.gz",
        "https://downloads.wortschatz-leipzig.de/corpora/deu-com_web_2021_1M.tar.gz",
        "https://downloads.wortschatz-leipzig.de/corpora/fra-ca_web_2020_300K.tar.gz",
        "https://downloads.wortschatz-leipzig.de/corpora/spa-ve_web_2016_300K.tar.gz",
    ]
):
    """Main function to strip the model and save it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the original tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    # Create the big model
    big_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    hidden_size = big_model.config.hidden_size

    # Download and process corpora
    corpus_dirs = download_and_extract_corpora(corpus_urls, data_dir="data")
    all_sentences = {}
    for lang, corpus_dir in corpus_dirs.items():
        sentences_files = find_sentences_files(corpus_dir)
        if sentences_files:
            sentences = read_sentences_with_fallback(sentences_files)
            if sentences:
                all_sentences[lang] = sentences
            else:
                print(f"Skipping language {lang}")
                continue

    counters_by_language = {}

    for lang, df in all_sentences.items():
        print(f"Processing language: {lang}")
        cnt = Counter()
        for text in tqdm(df, desc=f"Counting tokens for {lang}"):
            cnt.update(tokenizer(text)["input_ids"])
        counters_by_language[lang] = cnt
    
    
    resulting_vocab = {
        tokenizer.vocab[k] for k in tokenizer.special_tokens_map.values()
    }

    frequency_thresholds = {
        "en": 100,
        "ru": 2,
        "de": 10,
        "fr": 10,
        "es": 10,
    }

    low_id_threshold = 3000

    for lang, cnt in counters_by_language.items():
        print(f"Selecting tokens for language: {lang}")
        threshold = frequency_thresholds.get(lang, 0)  # Get threshold for the language, default to 0
        for k, v in cnt.items():
            if v >= threshold or k <= low_id_threshold:
                resulting_vocab.add(k)

    # Define the alphabets for your languages
    alphabets = {
        "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        "ru": "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        "de": "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞabcdefghijklmnopqrstuvwxyzäöüß",
        "fr": "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÂÇÉÈÊËÎÏÔŒÙÛÜŸabcdefghijklmnopqrstuvwxyzàâçéèêëîïôœùûüÿ",
        "es": "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑabcdefghijklmnopqrstuvwxyzáéíóúüñ",
    }

    # Find n-letter tokens
    n_letter_tokens = find_n_letter_tokens(tokenizer, alphabets, max_n=3)
    print(f"Found {len(n_letter_tokens)} n-letter tokens.")

    # Add n-letter tokens to resulting_vocab
    resulting_vocab.update(n_letter_tokens)

    resulting_vocab = sorted(list(resulting_vocab))

    print(f"Resulting vocabulary size: {len(resulting_vocab)}")
    print(f"Percentage of original vocab: {len(resulting_vocab) / tokenizer.vocab_size:.2f}")
    new_size = len(resulting_vocab)

    # --- Create the new vocabulary file with special tokens ---
    os.makedirs(new_model_name, exist_ok=True)
    inv_voc = {idx: word for word, idx in tokenizer.vocab.items()}

    # Get special tokens from the original tokenizer
    special_tokens_map = tokenizer.special_tokens_map
    print(f"Original special tokens map: {special_tokens_map}")

    with open(os.path.join(new_model_name, "vocab.txt"), "w", encoding="utf-8") as f:
        # Write special tokens first, maintaining order
        for token in special_tokens_map.get("all_special_tokens", []):
            f.write(token + "\n")

        # Write the rest of the tokens
        for idx in resulting_vocab:
            if idx in inv_voc and inv_voc[idx] not in special_tokens_map.get("all_special_tokens", []): # Exclude special tokens
                f.write(inv_voc[idx] + "\n")

    # Create the new embedding layer
    new_embedding = create_new_embedding(big_model, new_size, hidden_size, resulting_vocab, device)

    # Initialize a new model from the original model's config
    small_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # Replace the embeddings layer
    if isinstance(small_model, BertModel) or isinstance(small_model, transformers.BertForSequenceClassification):
        small_model.bert.set_input_embeddings(new_embedding)
    elif isinstance(small_model, RobertaModel) or isinstance(small_model, transformers.RobertaForSequenceClassification):
        small_model.roberta.set_input_embeddings(new_embedding)
    else:
        raise ValueError("Unsupported model type: must be Bert or Roberta")

    if big_model.config.tie_word_embeddings:
        small_model.tie_weights()

    print(f"Original model size: {msize(big_model):,}")
    print(f"New model size: {msize(small_model):,}")
    print(f"New Embedding layer size: {msize(small_model.bert.embeddings if isinstance(small_model, transformers.BertForSequenceClassification) else small_model.roberta.embeddings):,}")
    print(f"New Encoder layer size: {msize(small_model.bert.encoder if isinstance(small_model, transformers.BertForSequenceClassification) else small_model.roberta.encoder):,}")
    print(f"Size ratio new/old: {msize(small_model) / msize(big_model):.2f}")

    # Create a new tokenizer instance from the saved vocabulary
    new_tokenizer = transformers.BertTokenizerFast.from_pretrained(new_model_name)

    print(f'Supposedly saving new vocab_size ({new_size})')
    print(f'Old vocab_size: {small_model.config.vocab_size}')
    small_model.config.vocab_size = new_size
    print(f'New vocab_size: {small_model.config.vocab_size}')

    # Save the stripped model and the new tokenizer
    small_model.save_pretrained(new_model_name)
    new_tokenizer.save_pretrained(new_model_name)

    print("Saved stripped model.")

    # --- Check the new model and compare token IDs ---
    tokenizer = AutoTokenizer.from_pretrained(new_model_name) # Load the new tokenizer again for checking
    model = AutoModelForSequenceClassification.from_pretrained(new_model_name).to(device)

    text = "This is a test sentence in English and also a test предложение на русском."
    inputs_new = new_tokenizer(text, return_tensors="pt").to(device)
    inputs_old = AutoTokenizer.from_pretrained(model_name)(text, return_tensors="pt").to(device) # Load original tokenizer

    print("New Tokenizer Input IDs:", inputs_new['input_ids'])
    print("Old Tokenizer Input IDs:", inputs_old['input_ids'])

    # Inspect the new tokenizer's vocabulary
    print("New Tokenizer Vocabulary (first 50 entries):")
    for i in range(50):
      if i in tokenizer.vocab:
        print(f"  ID {i}: {tokenizer.vocab[i]}")

    texts = ["Это тестовое предложение на русском", "This is a test sentence in English"]
    e_new = embed(texts, model, new_tokenizer, device)  # Use new_tokenizer here
    e_old = embed(texts, big_model, AutoTokenizer.from_pretrained(model_name), device) # Load original tokenizer for embedding

    print(f"New embeddings shape: {e_new.shape}, old embeddings shape: {e_old.shape}")
    print(f"Cosine similarity (new model): {e_new[0].dot(e_new[1]):.4f}")
    print(f"Cosine similarity (old model): {e_old[0].dot(e_old[1]):.4f}")
    print(f"Cosine similarity between old and new: { (e_new * e_old).sum(1)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Strip a transformer model.")
    parser.add_argument("--model_name", type=str, default="./refusal_classifier_tiny", help="Name of the model to strip.")
    parser.add_argument("--new_model_name", type=str, default="refusal_classifier_tiny_stripped", help="Name of the model to save.")
    parser.add_argument("--vocab_threshold", type=int, default=5, help="Minimum token frequency.")
    parser.add_argument("--corpus_urls", type=str, nargs='+',
                        default=[
                            "https://downloads.wortschatz-leipzig.de/corpora/rus-ru_web-public_2019_1M.tar.gz",
                            "https://downloads.wortschatz-leipzig.de/corpora/eng-com_web-public_2018_1M.tar.gz",
                            "https://downloads.wortschatz-leipzig.de/corpora/deu-com_web_2021_1M.tar.gz",
                            "https://downloads.wortschatz-leipzig.de/corpora/fra-ca_web_2020_300K.tar.gz",
                            "https://downloads.wortschatz-leipzig.de/corpora/spa-ve_web_2016_300K.tar.gz",
                        ],
                        help="URLs of the corpora.")
    args = parser.parse_args()
    strip_model(
        model_name=args.model_name,
        new_model_name=args.new_model_name,
        vocab_threshold=args.vocab_threshold,
        corpus_urls=args.corpus_urls
    )
