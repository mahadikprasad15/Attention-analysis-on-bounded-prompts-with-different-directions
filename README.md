# Attention Analysis on Bounded Prompts

This repository implements a mechanistic interpretability experiment to analyze how LLMs encode harmfulness vs. refusal and how adversarial suffixes hijack attention mechanisms.

Based on:
- *LLMs Encode Harmfulness & Refusal Separately*
- *Universal Jailbreak Suffixes Are Strong Attention Hijackers*

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Log in to Hugging Face if using gated models (like Llama-3):
```bash
huggingface-cli login
```

## Usage

Run the complete pipeline:

```bash
python main.py
```

## Structure

- `src/config.py`: Configuration (model name, templates, suffixes).
- `src/data/`: Data loading and prompt formatting.
- `src/model/`: Activation collection hooks.
- `src/probes/`: Training linear probes for "Harmfulness" and "Refusal" directions.
- `src/evaluation/`: Baseline evaluation logic.
- `src/intervention/`: **Attention Intervention** logic (monkey-patching attention masks).
- `main.py`: Entry point.

## Intervention Logic

The `AttentionInterventionEvaluator` performs a "knockout" experiment. It modifies the attention mask during the forward pass to prevent the **Refusal Position** (start of response, `t_post`) from attending to the **Adversarial Suffix** tokens.

If the "Attention Hijack" hypothesis is correct, this intervention should restore the model's refusal behavior even in the presence of the jailbreak suffix.
