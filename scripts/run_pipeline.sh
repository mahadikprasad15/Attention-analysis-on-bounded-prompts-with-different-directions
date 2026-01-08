#!/bin/bash
set -e

# Default settings
MODEL="meta-llama/Llama-3.2-1B-Instruct"
SAVE_DIR="experiments/checkpoints"
SAMPLES=50

echo "Starting Experiment Pipeline..."
echo "Model: $MODEL"
echo "Save Dir: $SAVE_DIR"

# Step 1 & 2: Train Probes
echo "----------------------------------------"
echo "Phase 1: Training Probes"
echo "----------------------------------------"
python main.py \
    --model_name "$MODEL" \
    --step train \
    --save_dir "$SAVE_DIR" \
    --n_samples $SAMPLES

# Step 3, 4, 5, 6: Evaluation & Analysis
echo "----------------------------------------"
echo "Phase 2: Evaluation & Analysis"
echo "----------------------------------------"
python main.py \
    --model_name "$MODEL" \
    --step evaluate \
    --save_dir "$SAVE_DIR" \
    --n_samples $SAMPLES

echo "----------------------------------------"
echo "Pipeline Completed Successfully!"
echo "Results saved to intervention_results.png"
