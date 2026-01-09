
# Optimizing Adversarial Suffixes

You requested a way to find better adversarial suffixes because the current ones have high refusal rates.

## How to use the Optimizer

I created a new script: `src/experiments/optimize_suffix.py`.

It contains a list of known strong GCG suffixes and iterates through them, using your configured `RefusalLabeler` (LLM-based or pattern-based) to measure the success rate on a subset of harmful prompts.

### Run Command

```bash
!python src/experiments/optimize_suffix.py \
    --model_name "meta-llama/Llama-3.2-1B-Instruct" \
    --n_samples 20 \
    --use_llm_labeling \
    --cerebras_api_key "YOUR_KEY"
```

### Output
It will print a ranked list of suffixes. You can then copy the best ones (lowest refusal rate) and paste them into your `src/config.py` file under `adv_suffixes`.
