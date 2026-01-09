import torch
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

from src.config import REFUSAL_PHRASES
from src.data.formatting import PromptFormatter
from src.model.activations import ActivationCollector
from src.probes.linear import LinearProbe
from src.evaluation.refusal_labeler import RefusalLabeler

class ProbeTrainer:
    """Handles training probes and finding best layers"""

    def __init__(
        self,
        collector: ActivationCollector,
        formatter: PromptFormatter,
        use_llm_labeling: bool = False,
        llm_labeler_config: Dict = None
    ):
        self.collector = collector
        self.formatter = formatter

        # Initialize refusal labeler
        if llm_labeler_config is None:
            llm_labeler_config = {}
        self.refusal_labeler = RefusalLabeler(use_llm=use_llm_labeling, **llm_labeler_config)
    
    def train_harmfulness_probe(
        self,
        harmful_instructions: List[str],
        harmless_instructions: List[str],
        search_all_layers: bool = True
    ) -> Tuple[int, LinearProbe, Dict]:
        """
        Train harmfulness probe at t_inst position
        """
        print("=" * 60)
        print("TRAINING HARMFULNESS PROBE (at t_inst)")
        print("=" * 60)
        
        # Create prompts (no adversarial suffix)
        harmful_prompts = [
            self.formatter.create_prompt(instr, adv_suffix=None)
            for instr in harmful_instructions
        ]
        harmless_prompts = [
            self.formatter.create_prompt(instr, adv_suffix=None)
            for instr in harmless_instructions
        ]
        
        all_prompts = harmful_prompts + harmless_prompts
        labels = np.array(
            [1] * len(harmful_prompts) + [0] * len(harmless_prompts)
        )
        
        if search_all_layers:
            # Collect from all layers
            print(f"Collecting activations from all {self.collector.n_layers} layers...")
            activations_by_layer = self.collector.collect_all_layers(
                all_prompts, position_key='t_inst'
            )
            
            # Train probe on each layer
            results = {}
            best_acc = 0
            best_layer = None
            best_probe = None
            
            print("\nTraining probes:")
            for layer_idx in range(self.collector.n_layers):
                X = activations_by_layer[layer_idx].numpy()
                
                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=42
                )
                
                # Train
                probe = LinearProbe(self.collector.hidden_dim)
                train_acc = probe.fit(X_train, y_train)
                test_acc = probe.score(X_test, y_test)
                
                results[layer_idx] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'probe': probe
                }
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_layer = layer_idx
                    best_probe = probe
                
                if layer_idx % 4 == 0:  # Print every 4th layer
                    print(f"  Layer {layer_idx:2d}: Train={train_acc:.3f}, Test={test_acc:.3f}")
            
            print(f"\n✓ Best layer: {best_layer} (Test Acc={best_acc:.3f})")
            
        else:
            # Just use last layer
            print("Using last layer...")
            activations = self.collector.collect_single_layer(
                all_prompts, position_key='t_inst', layer_idx=-1
            )
            
            X = activations.numpy()
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )
            
            best_probe = LinearProbe(self.collector.hidden_dim)
            train_acc = best_probe.fit(X_train, y_train)
            test_acc = best_probe.score(X_test, y_test)
            
            best_layer = self.collector.n_layers - 1
            results = {best_layer: {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'probe': best_probe
            }}
            
            print(f"  Train={train_acc:.3f}, Test={test_acc:.3f}")
        
        return best_layer, best_probe, results
    
    def train_refusal_probe(
        self,
        harmful_instructions: List[str],
        harmless_instructions: List[str] = None, # Added optional harmless
        layer_idx: int = -1
    ) -> LinearProbe:
        """
        Train refusal probe at t_post position
        """
        print("\n" + "=" * 60)
        print("TRAINING REFUSAL PROBE (at t_post)")
        print("=" * 60)
        
        # Create harmful prompts (no adversarial suffix)
        harmful_prompts = [
            self.formatter.create_prompt(instr, adv_suffix=None)
            for instr in harmful_instructions
        ]
        
        # Create harmless prompts if provided (to ensure negative samples)
        harmless_prompts = []
        if harmless_instructions:
            harmless_prompts = [
                self.formatter.create_prompt(instr, adv_suffix=None)
                for instr in harmless_instructions
            ]
            
        all_prompts = harmful_prompts + harmless_prompts
        
        print(f"Generating outputs to label refusals ({len(harmful_prompts)} harmful, {len(harmless_prompts)} harmless)...")
        activations = []
        labels = []
        
        for i, prompt in enumerate(all_prompts):
            if i % 10 == 0:
                print(f"  Processing {i+1}/{len(all_prompts)}...")
            
            # Get activation at t_post
            act = self.collector.collect_single_layer(
                [prompt], position_key='t_post', layer_idx=layer_idx
            )[0]
            
            # For harmless prompts, we assume they are NOT refused (Class 0)
            # For harmful prompts, we generate and check
            is_harmful_input = i < len(harmful_prompts)
            
            if not is_harmful_input:
                # Harmless -> Not Refused
                refused = False
            else:
                # Harmful -> Generate and check
                inputs = self.collector.tokenizer(
                    prompt.text,
                    return_tensors='pt',
                    add_special_tokens=False
                ).to(self.collector.model.device)
                
                with torch.no_grad():
                    outputs = self.collector.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False
                    )
                
                generated = self.collector.tokenizer.decode(
                    outputs[0][len(inputs['input_ids'][0]):],
                    skip_special_tokens=True
                )
                
                refusal_result = self.refusal_labeler.check_refusal(generated, prompt.instruction)
                refused = refusal_result['refused']

            activations.append(act)
            labels.append(1 if refused else 0)
        
        activations = torch.stack(activations).numpy()
        labels = np.array(labels)
        
        print(f"\nRefusal rate: {labels.mean():.2%}")
        
        # Train probe
        X_train, X_test, y_train, y_test = train_test_split(
            activations, labels, test_size=0.2, random_state=42
        )
        
        probe = LinearProbe(self.collector.hidden_dim)
        train_acc = probe.fit(X_train, y_train)
        test_acc = probe.score(X_test, y_test)
        
        print(f"  Train={train_acc:.3f}, Test={test_acc:.3f}")
        
        return probe
    
    def _check_refusal(self, text: str) -> bool:
        """Check if generated text contains refusal"""
        text_lower = text.lower()
        first_part = text_lower[:100]  # Check first 100 chars
        return any(phrase.lower() in first_part for phrase in REFUSAL_PHRASES)
