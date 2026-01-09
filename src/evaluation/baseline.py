import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.config import REFUSAL_PHRASES
from src.data.structures import Prompt
from src.data.formatting import PromptFormatter
from src.probes.linear import LinearProbe
from src.evaluation.refusal_labeler import RefusalLabeler

@dataclass
class EvaluationResult:
    """Results from evaluating a single prompt"""
    instruction: str
    condition: str  # 'R', 'J', or 'F'
    has_adv: bool
    harm_score: float
    refusal_score: float
    actually_refuses: bool
    generated_text: str

class BaselineEvaluator:
    """Evaluates prompts on baseline (no intervention)"""

    def __init__(
        self,
        model,
        tokenizer,
        formatter: PromptFormatter,
        harm_probe: LinearProbe,
        refusal_probe: LinearProbe,
        harm_layer_idx: int,
        refusal_layer_idx: int = -1,
        use_llm_labeling: bool = False,
        llm_labeler_config: Dict = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.harm_probe = harm_probe
        self.refusal_probe = refusal_probe
        self.harm_layer_idx = harm_layer_idx
        self.refusal_layer_idx = refusal_layer_idx if refusal_layer_idx != -1 else len(model.model.layers) - 1

        # Initialize refusal labeler
        if llm_labeler_config is None:
            llm_labeler_config = {}
        self.refusal_labeler = RefusalLabeler(use_llm=use_llm_labeling, **llm_labeler_config)
    
    def evaluate_prompt(self, prompt: Prompt, condition: str) -> EvaluationResult:
        """
        Evaluate a single prompt
        
        Returns:
            EvaluationResult with all measurements
        """
        positions = self.formatter.get_positions(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt.text,
            return_tensors='pt'
        ).to(self.model.device)
        
        # Collect activations at both positions
        saved_activations = {}
        
        def make_hook(layer_idx, key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                saved_activations[key] = hidden_states.detach()
            return hook_fn
        
        # Hook both layers
        hook_harm = self.model.model.layers[self.harm_layer_idx].register_forward_hook(
            make_hook(self.harm_layer_idx, 'harm')
        )
        hook_ref = self.model.model.layers[self.refusal_layer_idx].register_forward_hook(
            make_hook(self.refusal_layer_idx, 'refusal')
        )
        
        try:
            # Generate
            # We want to catch the activations on the forward pass of generation
            # OR we can just do a forward pass first to get activations, then generate
            # Doing both in one go via generate is tricky because generate calls forward multiple times.
            # Ideally we want activations on the PREFILL step (first forward pass).
            
            # Efficient way:
            # 1. Forward pass for activations
            with torch.no_grad():
                _ = self.model(inputs['input_ids'])
            
            # remove hooks before generation to avoid capturing generation steps
            hook_harm.remove()
            hook_ref.remove()
            
            # Extract activations (from the prefill)
            act_t_inst = saved_activations['harm'][0, positions.t_inst, :].cpu()
            act_t_post = saved_activations['refusal'][0, positions.t_post, :].cpu()
            
            # 2. Generate for text output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False
                )
        
        finally:
            # safety net if something fails
            if hasattr(hook_harm, 'handle'): hook_harm.remove() # For older torch
            if hasattr(hook_ref, 'handle'): hook_ref.remove()
            
        
        # Get probe scores
        harm_score = self.harm_probe.get_continuous_score(act_t_inst)
        refusal_score = self.refusal_probe.get_continuous_score(act_t_post)
        
        # Decode generated text
        generated = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        
        # Check refusal using labeler
        refusal_result = self.refusal_labeler.check_refusal(generated, prompt.instruction)
        refused = refusal_result['refused']

        return EvaluationResult(
            instruction=prompt.instruction,
            condition=condition,
            has_adv=prompt.has_adv,
            harm_score=harm_score,
            refusal_score=refusal_score,
            actually_refuses=refused,
            generated_text=generated
        )
    
    def evaluate_dataset(
        self,
        prompts: List[Prompt],
        condition: str
    ) -> List[EvaluationResult]:
        """Evaluate multiple prompts"""
        results = []
        for i, prompt in enumerate(prompts):
            if i % 10 == 0:
                print(f"  Evaluating {i+1}/{len(prompts)}...")
            result = self.evaluate_prompt(prompt, condition)
            results.append(result)
        return results
    
    def _check_refusal(self, text: str) -> bool:
        """Check if text contains refusal"""
        text_lower = text.lower()
        first_part = text_lower[:100]
        return any(phrase.lower() in first_part for phrase in REFUSAL_PHRASES)
