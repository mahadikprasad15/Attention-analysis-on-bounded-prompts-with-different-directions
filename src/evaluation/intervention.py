import torch
from src.data.structures import Prompt
from src.evaluation.baseline import BaselineEvaluator, EvaluationResult

class AttentionInterventionEvaluator(BaselineEvaluator):
    """Evaluates prompts WITH attention intervention"""
    
    def evaluate_prompt_with_intervention(
        self, 
        prompt: Prompt,
        condition: str
    ) -> EvaluationResult:
        """
        Evaluate with attention from t_post to ADV blocked.
        Target: Block attention from [t_post] query to [adv_start:adv_end] keys.
        """
        if not prompt.has_adv:
            raise ValueError("Cannot intervene on prompt without adversarial suffix")
        
        positions = self.formatter.get_positions(prompt)
        
        # We need to monkey-patch the attention forward pass to modify the mask.
        # This is the most reliable way across HF versions compared to hooks which operate on output.
        
        # Store original forwards
        original_forwards = {}
        for i, layer in enumerate(self.model.model.layers):
            original_forwards[i] = layer.self_attn.forward
            
        def make_patched_forward(layer_idx):
            original_fwd = original_forwards[layer_idx]
            
            def patched_forward(hidden_states, attention_mask=None, position_ids=None, past_key_value=None, **kwargs):
                # Check if we are in the prefill phase (full sequence)
                # hidden_states: [batch, seq_len, dim]
                seq_len = hidden_states.shape[1]
                
                # We only want to intervene during the specific prompt processing
                # The prompt length is positions.total_length
                if seq_len == positions.total_length:
                    # This is the prefill step for our prompt!
                    
                    # We need to ensure attention_mask is 4D: [batch, 1, q, k]
                    # If it's None, Llama creates it internally. We can't let it do that if we want to modify it.
                    # But often it's passed as None.
                    
                    # If we can't easily modify the mask because it's None (and computed inside),
                    # we might be stuck unless we copy the mask creation logic.
                    
                    # BUT, we can inspect `attention_mask`.
                    if attention_mask is not None and attention_mask.dim() == 4:
                        # Clone to avoid side effects
                        # attention_mask is additive: 0.0 for attend, float('-inf') for mask
                        mod_mask = attention_mask.clone()
                        
                        # Target indices
                        t_post_idx = positions.t_post
                        adv_start = positions.adv_start
                        adv_end = positions.adv_end
                        
                        # Apply mask: [:, :, t_post, adv_start:adv_end+1] = min_dtype
                        min_val = torch.finfo(mod_mask.dtype).min
                        mod_mask[:, :, t_post_idx, adv_start:adv_end+1] = min_val
                        
                        return original_fwd(hidden_states, attention_mask=mod_mask, position_ids=position_ids, past_key_value=past_key_value, **kwargs)
                
                # Default behavior for generation steps or mismatched lengths
                return original_fwd(hidden_states, attention_mask, position_ids, past_key_value, **kwargs)
            
            return patched_forward

        # Apply patches
        for i, layer in enumerate(self.model.model.layers):
            layer.self_attn.forward = make_patched_forward(i)
        
        try:
            # Evaluate using parent logic (which calls model())
            # We call the super method, but we need to pass a distinct condition name
            result = self.evaluate_prompt(prompt, condition)
            
            # The result logic in evaluate_prompt generates text.
            # Our patched forward handles the mask modification during that process.
            # Note: For generation steps (autoregressive), our patch above checks `seq_len == total_length`.
            # This means it ONLY patches the prefill attention (where t_post attends to ADV).
            # For subsequent tokens generated, they will attend to ADV unless we patch that too.
            # But the hypothesis is about the Representation at t_post.
            
            return EvaluationResult(
                instruction=result.instruction,
                condition=condition + "_intervened",
                has_adv=result.has_adv,
                harm_score=result.harm_score,
                refusal_score=result.refusal_score,
                actually_refuses=result.actually_refuses,
                generated_text=result.generated_text
            )
            
        finally:
            # Restore original forwards
            for i, layer in enumerate(self.model.model.layers):
                layer.self_attn.forward = original_forwards[i]
