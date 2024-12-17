import os
import tarfile
import urllib.request
import urllib.error
from collections import Counter

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertModel, RobertaModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

def msize(model):
    """Calculates the number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())

def embed(sentences, model, tokenizer, device):
    """
    Generates sentence embeddings by accessing the last hidden state before the classification head.

    Parameters:
        sentences (list): A list of input sentences.
        model (torch.nn.Module): The classification model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        device (torch.device): The device to use (CPU or GPU).

    Returns:
        torch.Tensor: The sentence embeddings extracted from the last hidden state.
    """
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input, output_hidden_states=True)

        if isinstance(model, BertModel):
            # For Bert-like models, use the last hidden state
            embeddings = model_output.hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
        elif isinstance(model, RobertaModel):
            # For Roberta-like models, use the last hidden state
             embeddings = model_output.hidden_states[-1][:, 0, :]  # [batch_size, hidden_size]
        else:
          raise ValueError("Unsupported model type: must be Bert or Roberta")

        embeddings = torch.nn.functional.normalize(embeddings)

    return embeddings.cpu().numpy()

def download_and_extract_corpora(corpus_urls, corpus_dir_prefix="corpus"):
    """
    Downloads and extracts corpora from provided URLs.

    Parameters:
        corpus_urls (list): List of URLs for corpora.
        corpus_dir_prefix (str): Prefix for the directory names where corpora will be stored.

    Returns:
        dict: A dictionary where keys are language codes and values are directory paths to corpus data.
    """
    corpus_dirs = {}

    for url in tqdm(corpus_urls, desc="Processing corpus URLs"):
        filename = os.path.basename(url)
        language_code = filename.split("-")[0]  # Extract language code from filename
        corpus_file = f"{corpus_dir_prefix}_{language_code}.tar.gz"
        corpus_dir = filename.replace(".tar.gz", "")

        # Download the corpus if not already downloaded
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

        # Extract corpus if not already extracted
        if not os.path.exists(corpus_dir):
            print(f"Extracting '{corpus_file}'...")
            try:
                with tarfile.open(corpus_file, "r:gz") as tar:
                    tar.extractall()
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

    Parameters:
        corpus_dir (str): Path to the corpus directory.

    Returns:
        list: A list of paths to the sentences files.
    """
    sentence_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file_name in files:
            if any(
                keyword in file_name.lower()
                for keyword in ["sentences", "text", "sent"]
            ):
                file_path = os.path.join(root, file_name)
                print(f"Found sentences file: {file_path}")
                sentence_files.append(file_path)
    if not sentence_files:
        print(f"No sentences files found in '{corpus_dir}'.")
    return sentence_files

def read_sentences_with_fallback(file_paths, language):
    """
    Reads sentences from multiple files, handling potential encoding issues.

    Parameters:
        file_paths (list): List of paths to the sentences files.
        language (str): The language code of the corpus.

    Returns:
        list: A list of sentences, or None if reading fails.
    """
    all_sentences = []
    encodings = ['utf-8', 'windows-1251', 'latin-1']  # Try these encodings
    for file_path in file_paths:
        sentences = []
        for encoding in encodings:
             try:
                with open(file_path, "r", encoding=encoding) as f:
                    sentences.extend([line.strip() for line in f])
                    print(f"Successfully read {file_path} using {encoding} encoding.")
                    break # Encoding is found, stop
             except UnicodeDecodeError:
                continue # try another encoding
             except Exception as e:
                print(f"Error reading {file_path} with encoding {encoding}: {e}. Skipping.")
                continue # try another encoding
        else:
            print(f"Failed to read {file_path} with all tried encodings.")

        all_sentences.extend(sentences)

    if not all_sentences:
        return None
    return all_sentences

def create_new_embedding(old_model, new_vocab_size, hidden_size, selected_vocab, device):
    """
    Creates a new embedding layer with a specified vocabulary, copying weights from the old model.
    Handles both Bert and Roberta models.
    """

    if isinstance(old_model, BertModel):
         new_embedding = torch.nn.Embedding(
            num_embeddings=new_vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
        ).to(device)
         new_embedding.weight.data.copy_(old_model.embeddings.word_embeddings.weight[selected_vocab, :].to(device))
    elif isinstance(old_model, RobertaModel):
          new_embedding = torch.nn.Embedding(
              num_embeddings=new_vocab_size,
              embedding_dim=hidden_size,
              padding_idx=1,  # Roberta uses padding index 1
          ).to(device)
          new_embedding.weight.data.copy_(old_model.embeddings.word_embeddings.weight[selected_vocab, :].to(device))
    else:
        raise ValueError("Unsupported model type: must be Bert or Roberta")

    return new_embedding

def create_new_tokenizer(tokenizer, resulting_vocab, new_model_name, vocab_size = 80000):
    """
    Creates a new tokenizer based on an existing one, but uses a given vocab.
    """

    # Create a basic tokenizer from the original tokenizer (can be slow!)
    new_tokenizer = Tokenizer(models.BPE())
    new_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    new_tokenizer.decoder = decoders.ByteLevel()

    # Create the trainer for a byte level tokenizer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, # Set the vocab size
        special_tokens=list(tokenizer.special_tokens_map.values())
    )

    inv_voc = {idx: word for word, idx in tokenizer.vocab.items()}
    new_tokenizer.train_from_iterator(
        iterator = [inv_voc[idx] for idx in resulting_vocab],
        trainer=trainer
    )

    #Wrap the tokenizer in the Huggingface library
    wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object = new_tokenizer,
            model_max_length = tokenizer.model_max_length,
            bos_token = tokenizer.bos_token,
            eos_token = tokenizer.eos_token,
            unk_token = tokenizer.unk_token,
            sep_token = tokenizer.sep_token,
            pad_token = tokenizer.pad_token,
            cls_token = tokenizer.cls_token,
            mask_token = tokenizer.mask_token,
        )

    # Save the new tokenizer
    wrapped_tokenizer.save_pretrained(new_model_name)
    return wrapped_tokenizer

def main():
    """Main function to strip the model and save it."""
    MODEL_NAME = "./refusal_classifier_tiny"  # or "intfloat/multilingual-e5-small" if you start from that
    NEW_MODEL_NAME = "refusal_classifier_tiny_stripped"
    VOCAB_THRESHOLD = 5
    CORPUS_URLS = [
        "https://downloads.wortschatz-leipzig.de/corpora/rus-ru_web-public_2019_1M.tar.gz",  # Russian corpus
        "https://downloads.wortschatz-leipzig.de/corpora/eng-com_web-public_2018_1M.tar.gz",  # English corpus
        "https://downloads.wortschatz-leipzig.de/corpora/deu-com_web_2021_1M.tar.gz",  # German corpus
        "https://downloads.wortschatz-leipzig.de/corpora/fra-ca_web_2020_300K.tar.gz",  # French corpus
        "https://downloads.wortschatz-leipzig.de/corpora/spa-ve_web_2016_300K.tar.gz",  # Spanish corpus
    ]

    # Determine if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the original tokenizer and model config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)

    # Create the big model, transfer to device, and use it to generate the new embeddings
    big_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    hidden_size = big_model.config.hidden_size

    # Download and extract corpora
    corpus_dirs = download_and_extract_corpora(CORPUS_URLS)

    # Collect all the sentences from available languages.
    all_sentences = []
    for lang, corpus_dir in corpus_dirs.items():
        sentences_files = find_sentences_files(corpus_dir)
        if sentences_files:
            sentences = read_sentences_with_fallback(sentences_files, lang)
            if sentences:
                all_sentences.extend(sentences)
            else:
                print(f"Skipping language {lang}")
                continue

    # Token Frequency counting
    cnt = Counter()
    for text in tqdm(all_sentences, desc="Counting tokens"):
        cnt.update(tokenizer(text)["input_ids"])

    # Select high-frequency tokens
    resulting_vocab = {
        tokenizer.vocab[k] for k in tokenizer.special_tokens_map.values()
    }  # Add special tokens
    for k, v in cnt.items():
        if v >= VOCAB_THRESHOLD or k <= 3_000:
            resulting_vocab.add(k)  # Add frequent and low-id tokens

    resulting_vocab = sorted(resulting_vocab)
    print(f"Resulting vocabulary size: {len(resulting_vocab)}")
    print(f"Percentage of original vocab: {len(resulting_vocab) / tokenizer.vocab_size:.2f}")
    new_size = len(resulting_vocab)

    # Save the new vocabulary
    os.makedirs(NEW_MODEL_NAME, exist_ok=True)
    inv_voc = {idx: word for word, idx in tokenizer.vocab.items()}
    with open(os.path.join(NEW_MODEL_NAME, "vocab.txt"), "w", encoding="utf-8") as f:
        for idx in resulting_vocab:
            f.write(inv_voc[idx] + "\n")

    # Create the new embedding layer with the right vocab size
    new_embedding = create_new_embedding(big_model, new_size, hidden_size, resulting_vocab, device)

    # Initialize a new model from the original model's config, but replace the embeddings
    small_model = AutoModel.from_pretrained(MODEL_NAME).to(device) # Load the original model
    small_model.config.vocab_size = new_size # Update the vocab size in the config

    # Replace the embeddings layer with the new one for both Bert and Roberta
    if isinstance(small_model, BertModel):
        small_model.embeddings.word_embeddings = new_embedding
    elif isinstance(small_model, RobertaModel):
        small_model.embeddings.word_embeddings = new_embedding
    else:
        raise ValueError("Unsupported model type: must be Bert or Roberta")

    small_model.tie_weights()
    # small_model.to(device) # Model is already on the device

    print(f"Original model size: {msize(big_model):,}")
    print(f"New model size: {msize(small_model):,}")

    print(f"New Embedding layer size: {msize(small_model.embeddings):,}")
    print(f"New Encoder layer size: {msize(small_model.encoder):,}")

    print(f"Size ratio new/old: {msize(small_model) / msize(big_model):.2f}")

    # Create the new tokenizer from vocabulary
    new_tokenizer = create_new_tokenizer(tokenizer, resulting_vocab, NEW_MODEL_NAME, vocab_size = 80000)

    # Save the stripped model
    small_model.save_pretrained(NEW_MODEL_NAME)

    print("Saved stripped model.")

    # Check the new model (optional)
    tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_NAME)
    model = AutoModel.from_pretrained(NEW_MODEL_NAME).to(device)

    text = "This is a test sentence in English and also a test предложение на русском."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(inputs)

    texts = ["Это тестовое предложение на русском", "This is a test sentence in English"]
    e_new = embed(texts, model, tokenizer, device)

    #Load the old model for comparison.
    big_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    e_old = embed(texts, big_model, tokenizer, device)

    print(f"New embeddings shape: {e_new.shape}, old embeddings shape: {e_old.shape}")

    print(f"Cosine similarity (new model): {e_new[0].dot(e_new[1]):.4f}")
    print(f"Cosine similarity (old model): {e_old[0].dot(e_old[1]):.4f}")

    print(f"Cosine similarity between old and new: { (e_new * e_old).sum(1)}")

if __name__ == "__main__":
    main()
