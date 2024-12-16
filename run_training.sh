#!/bin/bash

# Set default values
MODEL_SIZE="xlm-roberta-base"
OUTPUT_DIR="./refusal_classifier"
BATCH_SIZE=64
EPOCHS=1
LR=1e-4
SEED=42

# Check if at least one data file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_file_1> [data_file_2 ...]"
    exit 1
fi

# Get data files as arguments
DATA_FILES=("$@")

# Run the Python script with the provided arguments
python train_classifier.py \
    --data_paths "${DATA_FILES[@]}" \
    --model_size "$MODEL_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --seed "$SEED"

