#!/bin/bash
set -e

# Quick test with fewer samples and CPU/MPS if needed
echo "Running quick test..."

python main.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --step all \
    --save_dir "experiments/test_checkpoints" \
    --n_samples 5

echo "Test complete!"
