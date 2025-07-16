#!/bin/bash

# CRSP Trajectory Seeding Script
# This script runs the trajectory seeding phase using the LIMO dataset

set -e

# Default configuration
CONFIG_FILE="seeding_config.yaml"
MODEL_PATH="~/models/deepseek-llm-7b-chat"
OUTPUT_DIR="./trajectory_seeding_output"
DATASET_PATH="GAIR/LIMO"
BATCH_SIZE=32
LEARNING_RATE=1e-5
EPOCHS=1
MAX_LENGTH=2048

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config CONFIG_FILE          Configuration file (default: seeding_config.yaml)"
            echo "  --model-path MODEL_PATH       Path to the model (default: ~/models/deepseek-llm-7b-chat)"
            echo "  --output-dir OUTPUT_DIR       Output directory (default: ./trajectory_seeding_output)"
            echo "  --dataset-path DATASET_PATH   LIMO dataset path (default: GAIR/LIMO)"
            echo "  --batch-size BATCH_SIZE       Training batch size (default: 32)"
            echo "  --learning-rate LEARNING_RATE Learning rate (default: 1e-5)"
            echo "  --epochs EPOCHS               Number of training epochs (default: 1)"
            echo "  --max-length MAX_LENGTH       Maximum sequence length (default: 2048)"
            echo "  --help                        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=== CRSP Trajectory Seeding Configuration ==="
echo "Config file: $CONFIG_FILE"
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Dataset path: $DATASET_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "Max length: $MAX_LENGTH"
echo "=============================================="

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if config file exists, if not create it with current parameters
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found, creating with current parameters..."
    cat > "$CONFIG_FILE" << EOF
# CRSP Trajectory Seeding Configuration
trajectory_seeding:
  enabled: true
  limo_dataset_path: "$DATASET_PATH"
  cache_dir: "~/.cache/crsp/limo"
  epochs: $EPOCHS
  learning_rate: $LEARNING_RATE
  batch_size: $BATCH_SIZE
  max_length: $MAX_LENGTH
  alpha: 0.1
  output_dir: "$OUTPUT_DIR"
  save_steps: 500
  logging_steps: 100
  weight_decay: 0.01
  warmup_steps: 100
  gradient_accumulation_steps: 1
  fp16: true
  dataloader_num_workers: 4
  validation:
    enabled: true
    eval_steps: 250
    eval_strategy: "steps"
    metric_for_best_model: "eval_loss"
    greater_is_better: false
    load_best_model_at_end: true
    save_total_limit: 3

# Model configuration
model:
  path: "$MODEL_PATH"
  trust_remote_code: true

# Tokenizer configuration  
tokenizer:
  path: "$MODEL_PATH"
  trust_remote_code: true
EOF
fi

# Run trajectory seeding
echo "Starting trajectory seeding..."
python -m rewarded_self_play.trajectory_seeding.run_seeding \
    --config "$CONFIG_FILE" \
    trajectory_seeding.limo_dataset_path="$DATASET_PATH" \
    trajectory_seeding.output_dir="$OUTPUT_DIR" \
    trajectory_seeding.batch_size="$BATCH_SIZE" \
    trajectory_seeding.learning_rate="$LEARNING_RATE" \
    trajectory_seeding.epochs="$EPOCHS" \
    trajectory_seeding.max_length="$MAX_LENGTH" \
    model.path="$MODEL_PATH"

echo "Trajectory seeding completed successfully!"
echo "Seeded model saved to: $OUTPUT_DIR"