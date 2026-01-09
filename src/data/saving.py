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
