import torch
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from src.data.formatting import PromptFormatter
from src.data.structures import Prompt

class AttentionAnalyzer:
    """Analyzes attention patterns in the model"""
    
    def __init__(self, model, tokenizer, formatter: PromptFormatter):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        
    def get_attention_contributions(
        self, 
        prompts: List[Prompt],
        layer_idx: int = -1
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Measure how much the model attends to Instruction vs Adv Suffix at t_post.
        
        Returns:
            Dictionary containing 'instruction_attn' and 'suffix_attn' lists (per head stats potentially)
            For simplicity, this version aggregates across all heads in the specified layer.
        """
        print(f"\nAnalyzing Attention Patterns for {len(prompts)} prompts...")
        
        results = {
            'instruction_attn': [],
            'suffix_attn': [],
            'system_attn': [] # Everything else (user_start, etc)
        }
        
        for i, prompt in enumerate(tqdm(prompts)):
            # Get positions
            # We assume the prompt is formatted correctly
            pos_info = self.formatter.get_positions(prompt)
            
            # Tokenize and Run
            inputs = self.tokenizer(
                prompt.text, 
                return_tensors='pt', 
                add_special_tokens=False
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model(
                    **inputs, 
                    output_attentions=True
                )
            
            # Get Attention: tuple(layers) -> [batch, heads, q_len, k_len]
            # We want specific layer, last token query
            # layer_attentions is a tuple of len(n_layers)
            # each element is [1, num_heads, seq_len, seq_len]
            
            # Select Layer
            layer_attn = outputs.attentions[layer_idx] # [1, H, S, S]
            
            # Select Query: t_post (last token)
            # shape -> [1, H, 1, S] -> squeeze -> [H, S]
            t_post_idx = pos_info.t_post
            attn_at_post = layer_attn[0, :, t_post_idx, :] # [H, S]
            
            # Aggregate across Heads (Average) -> [S]
            # We want to see total attention mass regardless of specific head specialization for now
            avg_attn = attn_at_post.mean(dim=0) # [S]
            
            # Sum masses for regions
            # 1. Instruction: [0 ... t_inst] (Rough approximation, might include start tokens)
            # Better: use explicit start if possible, but for now 0 to t_inst is "Instruction + Prefix"
            # Actually, let's try to be precise if we can. 
            # pos_info doesn't store start, but we can assume instruction starts after user_start.
            # But simpler: Instruction Mass = sum(0..t_inst) - (actually t_inst is INCLUSIVE final token of instr)
            
            # If suffix exists:
            # Suffix Mass = sum(adv_start .. adv_end)
            
            instr_mass = avg_attn[:pos_info.t_inst + 1].sum().item()
            
            suffix_mass = 0.0
            if prompt.has_adv:
                suffix_mass = avg_attn[pos_info.adv_start : pos_info.adv_end + 1].sum().item()
            
            # Update results
            results['instruction_attn'].append(instr_mass)
            results['suffix_attn'].append(suffix_mass)
            # System/Other is the rest (mostly special tokens)
            results['system_attn'].append(1.0 - instr_mass - suffix_mass)
            
        return results
