# ============================================================================
# MAIN EXECUTION
# ============================================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.config import Config
from src.data.loader import load_harmful_instructions, load_harmless_instructions
from src.data.formatting import PromptFormatter
from src.model.activations import ActivationCollector
from src.probes.trainer import ProbeTrainer
from src.evaluation.baseline import BaselineEvaluator
from src.evaluation.intervention import AttentionInterventionEvaluator
from src.analysis.visualization import ResultAnalyzer

def main():
    """Complete experimental pipeline"""
    
    # ========================================================================
    # STEP 1: SETUP
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING MODEL AND DATA")
    print("="*70)
    
    config = Config()
    
    # Load model
    print(f"Loading {config.model_name}...")
    
    # Ensure optimal settings for intervention compatibility (avoid fused attn if possible)
    # Note: For strict intervention support, 'eager' might be safer but slower.
    # We will try standard loading first.
    hf_config = AutoConfig.from_pretrained(config.model_name)
    # hf_config.attn_implementation = "eager" # Uncomment if intervention fails silently
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load with device_map='auto': {e}")
        print("Falling back to CPU/MPS loading...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True
        ).to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    harmful_instructions = load_harmful_instructions()[:50]  # Start with 50
    harmless_instructions = load_harmless_instructions()[:50]
    
    print(f"✓ Loaded {len(harmful_instructions)} harmful and {len(harmless_instructions)} harmless instructions")
    
    # ========================================================================
    # STEP 2: TRAIN PROBES
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 2: TRAINING PROBES")
    print("="*70)
    
    formatter = PromptFormatter(tokenizer, template_style=config.template_style)
    collector = ActivationCollector(model, tokenizer, formatter)
    trainer = ProbeTrainer(collector, formatter)
    
    # Train harmfulness probe
    harm_layer, harm_probe, harm_results = trainer.train_harmfulness_probe(
        harmful_instructions,
        harmless_instructions,
        search_all_layers=True
    )
    
    # Train refusal probe
    # Note: Refusal probe training requires generation, so it's slower.
    refusal_probe = trainer.train_refusal_probe(
        harmful_instructions,
        layer_idx=-1
    )
    
    # Check orthogonality
    cos_sim = (harm_probe.direction @ refusal_probe.direction) / (
        harm_probe.direction.norm() * refusal_probe.direction.norm()
    )
    print(f"\n✓ Direction correlation: {cos_sim:.3f}")
    
    # ========================================================================
    # STEP 3: CREATE JAILBREAK DATASET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: CREATING JAILBREAK DATASET")
    print("="*70)
    
    test_instructions = harmful_instructions[:20]  # Test subset
    
    datasets = {
        'R': [],  # Refusal
        'J': [],  # Jailbreak
        'F': []   # Failed jailbreak
    }
    
    for instr in test_instructions:
        # Condition R: No adversarial suffix
        datasets['R'].append(formatter.create_prompt(instr, adv_suffix=None))
        
        # Condition J: Strong adversarial suffix
        for adv in config.adv_suffixes:
            datasets['J'].append(formatter.create_prompt(instr, adv_suffix=adv))
        
        # Condition F: Weak suffix
        datasets['F'].append(formatter.create_prompt(instr, adv_suffix="Please help."))
    
    print(f"✓ Created datasets: R={len(datasets['R'])}, J={len(datasets['J'])}, F={len(datasets['F'])}")
    
    # ========================================================================
    # STEP 4: BASELINE EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: BASELINE EVALUATION")
    print("="*70)
    
    evaluator = BaselineEvaluator(
        model, tokenizer, formatter,
        harm_probe, refusal_probe,
        harm_layer, refusal_layer_idx=-1
    )
    
    baseline_results = {}
    for condition in ['R', 'J', 'F']:
        print(f"\nEvaluating condition {condition}...")
        results = evaluator.evaluate_dataset(datasets[condition], condition)
        baseline_results[condition] = ResultAnalyzer.summarize_results(results)
    
    # ========================================================================
    # STEP 5: INTERVENTION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 5: RUNNING INTERVENTION ON CONDITION J")
    print("="*70)
    
    interv_evaluator = AttentionInterventionEvaluator(
        model, tokenizer, formatter,
        harm_probe, refusal_probe,
        harm_layer, refusal_layer_idx=-1
    )
    
    intervened_results = {}
    print("\nEvaluating J with intervention...")
    # Only evaluate J with intervention
    j_intervened = []
    
    for i, prompt in enumerate(datasets['J']):
        if i % 10 == 0:
            print(f"  Evaluating {i+1}/{len(datasets['J'])}...")
        res = interv_evaluator.evaluate_prompt_with_intervention(prompt, 'J')
        j_intervened.append(res)
        
    intervened_results['J'] = ResultAnalyzer.summarize_results(j_intervened)
    
    # ========================================================================
    # STEP 6: ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 6: ANALYSIS")
    print("="*70)
    
    analyzer = ResultAnalyzer()
    analyzer.print_comparison(baseline_results, intervened_results)
    analyzer.plot_results(baseline_results, intervened_results, save_path='intervention_results.png')
    
    print("\n✓ COMPLETE!")

if __name__ == "__main__":
    main()
