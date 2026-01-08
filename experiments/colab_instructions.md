# Running on Google Colab

To run this project on Google Colab with Llama 3.2 1B (Gated Model), follow these steps.

## 1. Setup Environment
Create a new cell and run:

```python
# Install dependencies
!pip install -r requirements.txt
!pip install --upgrade transformers accelerate bitsandbytes
```

## 2. Authenticate with Hugging Face
Since Llama 3.2 is a gated model, you need to log in.

```python
from huggingface_hub import login
login()
# Enter your HF token when prompted (ensure it has Read access to the model)
```

## 3. Run the Pipeline
You can run the entire pipeline or step-by-step.

### Option A: Run Everything (Easiest)
```python
!python main.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --step all \
    --n_samples 50
```

### Option B: Step-by-Step (Best for debugging)

**Step 1: Train Probes**
```python
!python main.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --step train \
    --save_dir "experiments/colab_run" \
    --n_samples 50
```

**Step 2: Evaluate & Analyze**
```python
!python main.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --step evaluate \
    --save_dir "experiments/colab_run" \
    --n_samples 50
```

## Troubleshooting
- **Out of Memory (OOM)**: Reduce `--n_samples` or use a smaller model.
- **Login Error**: Ensure you have accepted the license terms for Llama 3.2 on the Hugging Face website.
- **Device Issues**: The script automatically picks CUDA if available. Verify with `!nvidia-smi`.
