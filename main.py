# ============================================================================
# MAIN EXECUTION
# ============================================================================
import torch
import argparse
import os
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src.config import Config
from src.data.loader import load_harmful_instructions, load_harmless_instructions
from src.data.formatting import PromptFormatter
from src.model.activations import ActivationCollector
from src.probes.trainer import ProbeTrainer
from src.evaluation.baseline import BaselineEvaluator
from src.evaluation.intervention import AttentionInterventionEvaluator
from src.analysis.visualization import ResultAnalyzer
from src.analysis.attention_pattern import AttentionAnalyzer
from src.data.saving import save_experiment_data, load_experiment_data

def save_probes(save_dir, harm_probe, refusal_probe, harm_layer):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'probes.pkl'), 'wb') as f:
        pickle.dump({
            'harm_probe': harm_probe,
            'refusal_probe': refusal_probe,
            'harm_layer': harm_layer
        }, f)
    print(f"Saved probes to {save_dir}/probes.pkl")

def load_probes(save_dir):
    path = os.path.join(save_dir, 'probes.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No probes found at {path}. Run --step train first.")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded probes from {path}")
    return data['harm_probe'], data['refusal_probe'], data['harm_layer']

def main():
    """Complete experimental pipeline"""
    parser = argparse.ArgumentParser(description="LLM Attention Analysis Pipeline")
    parser.add_argument("--model_name", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--step", type=str, default="all", choices=["all", "train", "evaluate"], help="Step to run")
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints", help="Directory to save/load probes")
    parser.add_argument("--n_samples", type=int, default=50, help="Number of samples to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument("--use_llm_labeling", action="store_true", help="Use LLM for refusal labeling")
    parser.add_argument("--cerebras_api_key", type=str, default=None, help="Cerebras API Key (or set env CEREBRAS_API_KEY)")
    parser.add_argument("--llm_model", type=str, default="llama3.1-8b", help="Model name for labeling")
    args = parser.parse_args()

    # Shared LLM Labeler Config
    llm_labeler_config = {
        "llm_provider": "cerebras",
        "api_key": args.cerebras_api_key,
        "model_name": args.llm_model
    }

    # ========================================================================
    # STEP 1: SETUP
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING MODEL AND DATA")
    print("="*70)
    
    config = Config()
    if args.model_name:
        config.model_name = args.model_name
    if args.device:
        config.device = args.device

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
            trust_remote_code=True,
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"Failed to load with device_map='auto': {e}")
        print("Falling back to CPU/MPS loading...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(config.device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    harmful_instructions = load_harmful_instructions()[:args.n_samples]
    harmless_instructions = load_harmless_instructions()[:args.n_samples]
    
    print(f"✓ Loaded {len(harmful_instructions)} harmful and {len(harmless_instructions)} harmless instructions")
    
    formatter = PromptFormatter(tokenizer, template_style=config.template_style)
    collector = ActivationCollector(model, tokenizer, formatter)

    harm_probe = None
    refusal_probe = None
    harm_layer = None

    # ========================================================================
    # STEP 2: TRAIN PROBES
    # ========================================================================
    if args.step in ["all", "train"]:
        print("\n" + "="*70)
        print("STEP 2: TRAINING PROBES")
        print("="*70)
        
        probes_path = os.path.join(args.save_dir, 'probes.pkl')
        if os.path.exists(probes_path):
            print(f"Found existing probes at {probes_path}. Loading...")
            harm_probe, refusal_probe, harm_layer = load_probes(args.save_dir)
            print("✓ Probes loaded from cache. Skipping training.")
        else:
            trainer = ProbeTrainer(
                collector, 
                formatter,
                use_llm_labeling=args.use_llm_labeling,
                llm_labeler_config=llm_labeler_config
            )
            
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
                harmless_instructions=harmless_instructions,
                layer_idx=-1
            )
            
            # Check orthogonality
            if harm_probe.direction is not None and refusal_probe.direction is not None:
                cos_sim = (harm_probe.direction @ refusal_probe.direction) / (
                    harm_probe.direction.norm() * refusal_probe.direction.norm()
                )
                print(f"\n✓ Direction correlation: {cos_sim:.3f}")
            
            # Save probes
            save_probes(args.save_dir, harm_probe, refusal_probe, harm_layer)

    if args.step == "train":
        print("\nTraining complete. Exiting...")
        return

    # Load probes if skipping training
    if args.step == "evaluate" and harm_probe is None:
        harm_probe, refusal_probe, harm_layer = load_probes(args.save_dir)

    # ========================================================================
    # STEP 3: CREATE JAILBREAK DATASET
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 3: CREATING JAILBREAK DATASET")
    print("="*70)
    
    print("="*70)
    
    datasets = {}
    activations = {}
    
    # Try loading first
    loaded_all = True
    for condition in ['R', 'J', 'F']:
        p, a = load_experiment_data(args.save_dir, condition)
        if p is None or a is None:
            loaded_all = False
            break
        datasets[condition] = p
        activations[condition] = a
        
    if loaded_all:
        print("✓ Loaded all datasets and activations from cache. Skipping generation.")
        acts_R = activations['R']
        acts_J = activations['J']
        acts_F = activations['F']
    else:
        # Use a subset for testing (e.g., 40% of loaded samples)
        test_n = max(1, int(len(harmful_instructions) * 0.4))
        test_instructions = harmful_instructions[:test_n]
        print(f"Using {len(test_instructions)} samples for evaluation")
        
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
        
        # SAVE DATASETS AND ACTIVATIONS
        print("\nSaving experiment data...")
        # 1. Condition R (Clean)
        print("  Collecting activations for R...")
        acts_R = collector.collect_single_layer(datasets['R'], 't_post', layer_idx=-1)
        save_experiment_data(args.save_dir, 'R', datasets['R'], acts_R)
        
        # 2. Condition J (Jailbreak)
        print("  Collecting activations for J...")
        acts_J = collector.collect_single_layer(datasets['J'], 't_post', layer_idx=-1)
        save_experiment_data(args.save_dir, 'J', datasets['J'], acts_J)
        
        # 3. Condition F (Failed)
        print("  Collecting activations for F...")
        acts_F = collector.collect_single_layer(datasets['F'], 't_post', layer_idx=-1)
        save_experiment_data(args.save_dir, 'F', datasets['F'], acts_F)
    
    # ========================================================================
    # STEP 4: BASELINE EVALUATION
    # ========================================================================
    print("\n" + "="*70)
    print("STEP 4: BASELINE EVALUATION")
    print("="*70)
    
    evaluator = BaselineEvaluator(
        model, tokenizer, formatter,
        harm_probe, refusal_probe,
        harm_layer, refusal_layer_idx=-1,
        use_llm_labeling=args.use_llm_labeling,
        llm_labeler_config=llm_labeler_config
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
        harm_layer, refusal_layer_idx=-1,
        use_llm_labeling=args.use_llm_labeling,
        llm_labeler_config=llm_labeler_config
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
    analyzer.plot_results(baseline_results, intervened_results, save_path=os.path.join(args.save_dir, 'intervention_results.png'))
    
    # ATTENTION ANALYSIS
    print("\n" + "="*70)
    print("ANALYZING ATTENTION PATTERNS (Jailbreak)")
    print("="*70)
    
    
    attn_analyzer = AttentionAnalyzer(model, tokenizer, formatter)
    # Analyze Condition J (Jailbreak) where suffixes are present
    attn_results = attn_analyzer.get_attention_contributions(datasets['J'], layer_idx=-1) # Last layer
    
    # Visualize Overall
    analyzer.plot_attention_contributions(attn_results, save_path=os.path.join(args.save_dir, 'attention_analysis.png'))
    
    # GRANULAR COMPARISON (Refused vs Complied after Intervention)
    print("\n" + "="*70)
    print("ANALYZING ATTENTION: REFUSED vs COMPLIED")
    print("="*70)
    
    # 1. Identify "Refused" vs "Complied" from INTERVENTION results for 'J'
    # We need the actual prompts corresponding to these outcomes
    # intervened_results['J'] is a summary. We need the list 'j_intervened' from step 5.
    # To do this cleanly without refactoring too much, let's look at 'j_intervened' list.
    # But scope of j_intervened is local to loop.
    # We need to make j_intervened available here.
    # Simplest way: Re-generate or move logic up.
    # Wait, 'j_intervened' is available if we are in the same scope, but previous code didn't save it.
    
    # Let's verify if j_intervened is accessible. In the previous code, j_intervened was a local list 
    # inside the Step 5 block. If Step 5 and 6 are in 'main', and j_intervened is defined in main, it works.
    # Looking at file content... yes, it is in main().
    
    if 'j_intervened' in locals():
        # Sort by refusal score (higher = more likely refused)
        # Or filter by binary 'actually_refuses'
        
        refused_samples = [res for res in j_intervened if res.actually_refuses]
        complied_samples = [res for res in j_intervened if not res.actually_refuses]
        
        print(f"Found {len(refused_samples)} Refused and {len(complied_samples)} Complied samples in Condition J (Intervened).")
        
        if len(refused_samples) > 0 and len(complied_samples) > 0:
            # Sort by refusal score confidence
            refused_samples.sort(key=lambda x: x.refusal_score, reverse=True) # Highest refusal score
            complied_samples.sort(key=lambda x: x.refusal_score) # Lowest refusal score (most compliant)
            
            # Take Top 3
            top_refused = refused_samples[:3]
            top_complied = complied_samples[:3]
            
            # Extract Prompts
            # We need to map EvaluationResult back to Prompt object
            # Or just recreate text? AttentionAnalyzer needs Prompt object or text.
            # AttentionAnalyzer takes List[Prompt].
            # Warning: EvaluationResult only has text strings.
            # We need to find the original Prompt object from datasets['J'] that matches.
            # This is O(N^2) but N=50 so it's fine.
            
            def get_prompts_from_results(results_list, all_prompts):
                target_prompts = []
                for res in results_list:
                    # Find matching prompt in all_prompts
                    for p in all_prompts:
                        if p.instruction == res.instruction: # Match by instruction
                            target_prompts.append(p)
                            break
                return target_prompts

            prompts_refused = get_prompts_from_results(top_refused, datasets['J'])
            prompts_complied = get_prompts_from_results(top_complied, datasets['J'])
            
            # Run Attention Analysis
            print("Analyzing Top 3 Refused...")
            attn_refused = attn_analyzer.get_attention_contributions(prompts_refused, layer_idx=-1)
            
            print("Analyzing Top 3 Complied...")
            attn_complied = attn_analyzer.get_attention_contributions(prompts_complied, layer_idx=-1)
            
            # Plot Comparison
            analyzer.plot_attention_breakdown_comparison(
                attn_refused, 
                attn_complied, 
                save_path=os.path.join(args.save_dir, 'attention_breakdown_comparison.png')
            )
        else:
            print("Skipping comparison: Need at least one Refused and one Complied sample.")
            
    else:
        print("Warning: j_intervened results not found. Skipping granular analysis.")
    
    print("\n✓ COMPLETE!")

if __name__ == "__main__":
    main()
