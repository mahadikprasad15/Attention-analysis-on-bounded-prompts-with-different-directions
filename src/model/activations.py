import torch
from typing import List, Dict, Optional
from src.data.structures import Prompt
from src.data.formatting import PromptFormatter

class ActivationCollector:
    """Collects activations from model at specific positions"""
    
    def __init__(self, model, tokenizer, formatter: PromptFormatter):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = formatter
        self.n_layers = len(model.model.layers)
        self.hidden_dim = model.config.hidden_size
    
    def collect_single_layer(
        self, 
        prompts: List[Prompt], 
        position_key: str,  # 't_inst' or 't_post'
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Collect activations at a specific position from one layer
        
        Args:
            prompts: List of Prompt objects
            position_key: Which position to extract ('t_inst' or 't_post')
            layer_idx: Which layer (-1 for last layer)
        
        Returns:
            activations: [n_prompts, hidden_dim]
        """
        if layer_idx == -1:
            layer_idx = self.n_layers - 1
        
        activations = []
        
        for prompt in prompts:
            # Get token positions
            positions = self.formatter.get_positions(prompt)
            target_pos = getattr(positions, position_key)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt.text, 
                return_tensors='pt'
            ).to(self.model.device)
            
            # Storage for this layer's activation
            saved_activation = {}
            
            def hook_fn(module, input, output):
                # output[0] is hidden states: [batch, seq_len, hidden_dim]
                saved_activation['hidden'] = output[0].detach()
            
            # Register hook
            hook = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
            
            # Forward pass
            try:
                with torch.no_grad():
                    _ = self.model(**inputs)
            finally:
                # Remove hook
                hook.remove()
            
            # Extract at target position
            # Note: target_pos is int, assume batch size 1
            act = saved_activation['hidden'][0, target_pos, :].cpu()
            activations.append(act)
        
        return torch.stack(activations)
    
    def collect_all_layers(
        self,
        prompts: List[Prompt],
        position_key: str
    ) -> Dict[int, torch.Tensor]:
        """
        Collect activations from all layers at a specific position
        
        Args:
            prompts: List of Prompt objects
            position_key: Which position to extract ('t_inst' or 't_post')
        
        Returns:
            activations_by_layer: {layer_idx: [n_prompts, hidden_dim]}
        """
        activations_by_layer = {i: [] for i in range(self.n_layers)}
        
        for prompt in prompts:
            positions = self.formatter.get_positions(prompt)
            target_pos = getattr(positions, position_key)
            
            inputs = self.tokenizer(
                prompt.text,
                return_tensors='pt'
            ).to(self.model.device)
            
            # Storage for all layers
            layer_activations = {}
            hooks = []
            
            # Register hooks on all layers
            for layer_idx in range(self.n_layers):
                def make_hook(idx):
                    def hook_fn(module, input, output):
                        layer_activations[idx] = output[0].detach()
                    return hook_fn
                
                hook = self.model.model.layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx)
                )
                hooks.append(hook)
            
            # Forward pass
            try:
                with torch.no_grad():
                    _ = self.model(**inputs)
            finally:
                # Remove all hooks
                for hook in hooks:
                    hook.remove()
            
            # Extract from each layer
            for layer_idx in range(self.n_layers):
                act = layer_activations[layer_idx][0, target_pos, :].cpu()
                activations_by_layer[layer_idx].append(act)
        
        # Stack each layer's activations
        for layer_idx in range(self.n_layers):
            activations_by_layer[layer_idx] = torch.stack(
                activations_by_layer[layer_idx]
            )
        
        return activations_by_layer
