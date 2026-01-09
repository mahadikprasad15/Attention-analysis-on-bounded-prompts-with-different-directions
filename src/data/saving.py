import os
import torch
import json
from typing import List
from src.data.structures import Prompt

def save_experiment_data(
    save_dir: str, 
    name: str, 
    prompts: List[Prompt], 
    activations: torch.Tensor
):
    """
    Save experiment data (prompts and activations) to disk.
    
    Args:
        save_dir: Directory to save files in
        name: Name prefix for files (e.g., 'J', 'R')
        prompts: List of Prompt objects
        activations: Tensor of activations [batch, hidden_dim]
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Prompts as JSON
    prompts_data = [
        {
            "text": p.text,
            "instruction": p.instruction,
            "has_adv": p.has_adv,
            "adv_suffix": p.adv_suffix
        }
        for p in prompts
    ]
    
    prompts_path = os.path.join(save_dir, f"{name}_prompts.json")
    with open(prompts_path, 'w') as f:
        json.dump(prompts_data, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {prompts_path}")
    
    # Save Activations as PT
    # Ensure activations are on CPU before saving
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu()
        
    acts_path = os.path.join(save_dir, f"{name}_activations.pt")
    torch.save(activations, acts_path)
    print(f"Saved activations shape {activations.shape} to {acts_path}")

def load_experiment_data(save_dir: str, name: str):
    """
    Load experiment data (prompts and activations) from disk.
    Returns (prompts, activations) or (None, None) if not found.
    """
    prompts_path = os.path.join(save_dir, f"{name}_prompts.json")
    acts_path = os.path.join(save_dir, f"{name}_activations.pt")
    
    if not (os.path.exists(prompts_path) and os.path.exists(acts_path)):
        return None, None
        
    print(f"Loading {name} data from {save_dir}...")
    
    # Load Prompts
    with open(prompts_path, 'r') as f:
        prompts_data = json.load(f)
        
    prompts = [
        Prompt(
            text=p['text'],
            instruction=p['instruction'],
            has_adv=p['has_adv'],
            adv_suffix=p.get('adv_suffix'),
            template_style="llama3" # Defaulting or need to save this too?
        )
        for p in prompts_data
    ]
    
    # Load Activations
    activations = torch.load(acts_path)
    
    print(f"✓ Loaded {len(prompts)} prompts and activations {activations.shape}")
    return prompts, activations

import pickle

def save_results(save_dir: str, name: str, results: list):
    """Save list of EvaluationResult objects"""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}_results.pkl")
    with open(path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved {len(results)} results to {path}")

def load_results(save_dir: str, name: str) -> list:
    """Load list of EvaluationResult objects"""
    path = os.path.join(save_dir, f"{name}_results.pkl")
    if not os.path.exists(path):
        return None
    
    with open(path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} results from {path}")
    return results
