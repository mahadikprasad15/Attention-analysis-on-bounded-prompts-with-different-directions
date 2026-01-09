import torch
from src.data.structures import Prompt
from src.evaluation.baseline import BaselineEvaluator, EvaluationResult

class AttentionInterventionEvaluator(BaselineEvaluator):
    """Evaluates prompts WITH attention intervention"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervention_applied = False  # Track if intervention actually happens
        self.intervention_count = 0  # Count how many times patch was triggered

    def evaluate_prompt_with_intervention(
        self,
        prompt: Prompt,
        condition: str,
        debug: bool = False
    ) -> EvaluationResult:
        """
        Evaluate with attention from t_post to ADV blocked.
        Target: Block attention from [t_post] query to [adv_start:adv_end] keys.
        """
        if not prompt.has_adv:
            raise ValueError("Cannot intervene on prompt without adversarial suffix")

        positions = self.formatter.get_positions(prompt, debug=debug)

        # Reset intervention tracking
        self.intervention_applied = False
        self.intervention_count = 0

        if debug:
            print("\n" + "="*80)
            print("INTERVENTION DEBUG")
            print("="*80)
            print(f"Target positions:")
            print(f"  t_post (query): {positions.t_post}")
            print(f"  ADV suffix (keys): [{positions.adv_start}:{positions.adv_end}]")
            print(f"  Total length: {positions.total_length}")

        # Store original forwards
        original_forwards = {}
        for i, layer in enumerate(self.model.model.layers):
            original_forwards[i] = layer.self_attn.forward

        def make_patched_forward(layer_idx):
            original_fwd = original_forwards[layer_idx]

            def patched_forward(hidden_states, attention_mask=None, position_ids=None, **kwargs):
                # Check if we are in the prefill phase (full sequence)
                # hidden_states: [batch, seq_len, dim]
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]

                # We only want to intervene during the specific prompt processing
                # Use exact match, but also track what seq_lens we see for debugging
                if layer_idx == 0 and debug:
                    if not hasattr(self, '_seen_seq_lens'):
                        self._seen_seq_lens = set()
                    self._seen_seq_lens.add(seq_len)

                if seq_len == positions.total_length:
                    # This is the prefill step for our prompt!
                    self.intervention_count += 1

                    # CRITICAL: Ensure we have a 4D attention mask
                    # Format: [batch, num_heads, query_len, key_len]
                    # For causal attention: mask[q, k] = 0 if q >= k, else -inf

                    if attention_mask is None:
                        # Create causal mask manually
                        if debug and layer_idx == 0:
                            print(f"\n[Layer {layer_idx}] Creating causal mask (was None)")

                        # Create 4D causal mask
                        device = hidden_states.device
                        dtype = hidden_states.dtype

                        # Causal mask: upper triangle is -inf
                        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
                        mask = torch.triu(mask, diagonal=1)  # Upper triangle (future tokens)
                        # Expand to [batch, 1, seq_len, seq_len]
                        attention_mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)

                    elif attention_mask.dim() == 2:
                        # Convert 2D padding mask to 4D attention mask
                        if debug and layer_idx == 0:
                            print(f"\n[Layer {layer_idx}] Converting 2D mask to 4D")

                        # attention_mask is [batch, seq_len] with 1=attend, 0=mask
                        # Convert to [batch, 1, 1, seq_len]
                        inverted_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                        # Convert 1/0 to 0/-inf
                        inverted_mask = (1.0 - inverted_mask) * torch.finfo(hidden_states.dtype).min

                        # Create causal mask and combine
                        causal_mask = torch.full((seq_len, seq_len), torch.finfo(hidden_states.dtype).min, device=hidden_states.device)
                        causal_mask = torch.triu(causal_mask, diagonal=1)
                        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                        # Combine: both padding and causal
                        attention_mask = causal_mask + inverted_mask

                    elif attention_mask.dim() == 4:
                        # Already 4D, just use it
                        if debug and layer_idx == 0:
                            print(f"\n[Layer {layer_idx}] Mask already 4D: {attention_mask.shape}")
                    else:
                        raise ValueError(f"Unexpected attention_mask dimension: {attention_mask.dim()}")

                    # Now we definitely have a 4D mask - clone and modify
                    mod_mask = attention_mask.clone()

                    # Target indices
                    t_post_idx = positions.t_post
                    adv_start = positions.adv_start
                    adv_end = positions.adv_end

                    # INTERVENTION: Block attention from t_post to ADV tokens
                    # Set mask[:, :, t_post, adv_start:adv_end+1] = -inf
                    min_val = torch.finfo(mod_mask.dtype).min
                    mod_mask[:, :, t_post_idx, adv_start:adv_end+1] = min_val

                    if layer_idx == 0:  # Only mark once per forward pass
                        self.intervention_applied = True

                    if debug and layer_idx == 0:
                        print(f"\n[Layer {layer_idx}] INTERVENTION APPLIED")
                        print(f"  Blocked attention from position {t_post_idx} to positions [{adv_start}:{adv_end}]")
                        print(f"  Mask shape: {mod_mask.shape}")
                        # Verify intervention
                        blocked_values = mod_mask[0, 0, t_post_idx, adv_start:adv_end+1]
                        print(f"  Blocked values: {blocked_values[:5].tolist()} (should be {min_val})")

                    # Call original forward with modified mask
                    return original_fwd(hidden_states, attention_mask=mod_mask, position_ids=position_ids, **kwargs)

                # Default behavior for generation steps or mismatched lengths
                return original_fwd(hidden_states, attention_mask=attention_mask, position_ids=position_ids, **kwargs)

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

            # VERIFICATION: Ensure intervention actually happened
            if not self.intervention_applied:
                seen_lens = getattr(self, '_seen_seq_lens', set())
                raise RuntimeError(
                    f"INTERVENTION FAILED! Patch was never applied.\n"
                    f"  Intervention count: {self.intervention_count}\n"
                    f"  Expected total_length: {positions.total_length}\n"
                    f"  Actual seq_lens seen: {sorted(seen_lens)}\n"
                    f"  This likely means:\n"
                    f"    - Tokenization mismatch (check add_special_tokens consistency)\n"
                    f"    - Model internals changed the sequence length\n"
                    f"    - Batching issues"
                )

            if debug:
                print(f"\n✓ Intervention verified: Applied {self.intervention_count} times")
                print("="*80 + "\n")

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
