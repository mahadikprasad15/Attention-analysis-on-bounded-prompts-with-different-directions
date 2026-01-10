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
            'system_attn': [],  # Template tokens and other system tokens
            'user_prefix_attn': []  # user_start template tokens
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
            
            # Sum masses for regions with CORRECT boundaries
            # 1. User prefix (template): [0 ... inst_start-1]
            # 2. Instruction: [inst_start ... t_inst]
            # 3. Adversarial suffix: [adv_start ... adv_end] (if present)
            # 4. System/Other: Everything else (user_end, assistant_start)

            # User prefix (template tokens like <|start_header_id|>user<|end_header_id|>)
            user_prefix_mass = avg_attn[:pos_info.inst_start].sum().item() if pos_info.inst_start > 0 else 0.0

            # Instruction (actual instruction text only)
            instr_mass = avg_attn[pos_info.inst_start : pos_info.t_inst + 1].sum().item()

            # Adversarial suffix (if present)
            suffix_mass = 0.0
            if prompt.has_adv:
                suffix_mass = avg_attn[pos_info.adv_start : pos_info.adv_end + 1].sum().item()

            # System/Other is the rest (user_end, assistant_start)
            system_mass = 1.0 - user_prefix_mass - instr_mass - suffix_mass

            # Update results
            results['instruction_attn'].append(instr_mass)
            results['suffix_attn'].append(suffix_mass)
            results['user_prefix_attn'].append(user_prefix_mass)
            results['system_attn'].append(system_mass)
            
        return results

        return detailed_results

    def get_head_contributions(
        self,
        prompts: List[Prompt],
        refusal_direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the contribution of each attention head to the refusal direction.
        
        Contribution = (HeadOutput @ W_O) @ RefusalDirection
        
        Args:
            prompts: List of Prompt objects
            refusal_direction: Normalized refusal direction vector [hidden_dim]
            
        Returns:
            contributions: [n_layers, n_heads] matrix of average scores
        """
        print(f"\nCalculating Head Contributions for {len(prompts)} prompts...")
        
        n_layers = self.model.config.num_hidden_layers
        n_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // n_heads
        
        # Initialize storage for running average
        total_scores = torch.zeros(n_layers, n_heads, device=self.model.device)
        
        # Ensure direction is on device
        refusal_direction = refusal_direction.to(self.model.device)
        
        for prompt in tqdm(prompts):
            pos_info = self.formatter.get_positions(prompt)
            t_post = pos_info.t_post
            
            inputs = self.tokenizer(
                prompt.text,
                return_tensors='pt',
                add_special_tokens=False
            ).to(self.model.device)
            
            # We need to capture the output of each head *before* W_O, 
            # OR capture the output *after* W_O but separated by head.
            # Most HF models don't expose separated W_O outputs easily via hooks 
            # without re-implementing the attention forward pass or hooking into internal submodules.
            
            # Strategy:
            # 1. Capture 'attn_output' (after W_O) -> No, that's mixed.
            # 2. Capture 'v_states' and 'attn_weights' to reconstruct? Expensive.
            # 3. Use 'output_attentions=True' to get weights (A).
            #    We need Value (V). V is usually not returned.
            
            # Alternative: Hook into 'self_attn' module.
            # In Llama, self_attn returns (attn_output, ...).
            # attn_output is [batch, seq, hidden].
            # This is already summed over heads.
            
            # We must compute z = Softmax(...) @ V.
            # Then contribution = (z @ W_O) @ dir = z @ (W_O @ dir).
            
            # To avoid insane complexity, let's use a standard approximation or specific structure assumption.
            # LlamaAttention:
            #   q, k, v = proj(x)
            #   attn = softmax(q @ k.T)
            #   out = attn @ v
            #   final = out @ o_proj
            
            # We can capture 'out' (concatenated head outputs) if we hook internal logic, 
            # OR we can manually compute it if we have inputs.
            
            # Simpler approach for "Probe Analysis":
            # For each layer, run the model.
            # Capture input to self_attn? No.
            
            # Let's try to capture 'v_proj(x)' and 'attn_weights' (from output_attentions).
            # But we don't have v_proj outputs easily.
            
            # OK, actually, many interpretability libraries do this. 
            # For this context, let's assume we can hook 'o_proj'.
            # Input to o_proj is [batch, seq, hidden]. 
            # Wait, in Llama, 'o_proj' maps hidden -> hidden?
            # No, 'o_proj' maps (n_heads * head_dim) -> hidden.
            # If we hook the input to `o_proj`, we get the concatenated head outputs!
            # Shape: [batch, seq_len, n_heads * head_dim]
            
            # PERFECT.
            
            layer_head_outputs = {}
            
            def get_o_input_hook(layer_idx):
                def hook(module, input, output):
                    # input[0] is the tensor passed to o_proj
                    # Shape: [batch, seq, hidden_dim] 
                    # (Note: hidden_dim = n_heads * head_dim)
                    # We only care about t_post
                    layer_head_outputs[layer_idx] = input[0][0, t_post, :].detach()
                return hook

            # Register hooks on o_proj
            hooks = []
            for i in range(n_layers):
                # Access submodule safely. For Llama: model.layers[i].self_attn.o_proj
                layer_module = self.model.model.layers[i]
                if hasattr(layer_module, 'self_attn'):
                    hooks.append(layer_module.self_attn.o_proj.register_forward_hook(get_o_input_hook(i)))
            
            # Run Model
            with torch.no_grad():
                self.model(**inputs)
                
            # Remove hooks
            for h in hooks: h.remove()
            
            # Process this prompt
            for i in range(n_layers):
                if i not in layer_head_outputs: continue
                
                # Full concatenated output: [n_heads * head_dim]
                concat_z = layer_head_outputs[i]
                
                # Reshape to [n_heads, head_dim]
                z_per_head = concat_z.view(n_heads, head_dim)
                
                # Get W_O for this layer
                W_O = self.model.model.layers[i].self_attn.o_proj.weight # [hidden, hidden] (transposed in Linear?)
                # Linear layer: y = x @ A.T + b. 
                # .weight is [out_features, in_features] -> [hidden, hidden]
                # So we want effective W_O part for each head.
                
                # W_O splits into heads along the *input* dimension (columns).
                # W_O = [W_O_h1, W_O_h2, ...]
                # W_O.weight is [hidden_dim, hidden_dim].
                # Reshape weight to [hidden_dim, n_heads, head_dim]
                W_O_reshaped = W_O.view(self.model.config.hidden_size, n_heads, head_dim)
                
                # We want contribution = (z_h @ W_O_h.T) @ dir
                # Actually simpler:
                # Total output = sum_h (z_h @ W_O_h.T)
                # Projection = Total @ dir = sum_h (z_h @ W_O_h.T @ dir)
                # Contribution_h = z_h @ W_O_h.T @ dir
                
                # Let's clean up shapes:
                # z_per_head: [heads, head_dim]
                # W_O_reshaped: [hidden, heads, head_dim]
                # Refusal dir: [hidden]
                
                # Precompute (W_O @ dir) part?
                # Project refusal direction back through W_O?
                # v_proj = dir @ W_O_reshaped -> [heads, head_dim]
                #   (hidden @ [hidden, heads, head_dim]) sum over hidden
                #   = einsum('d, dhs -> hs', dir, W_O_reshaped)
                
                # This vector 'v_proj' represents the direction in *head output space* that aligns with refusal.
                
                # Compute effective direction for each head
                # W_O weight is [out, in] = [hidden, heads*head_dim]
                # reshaped: [hidden, n_heads, head_dim]
                
                # Using einsum for clarity:
                # dir: d (hidden)
                # W: d n h (hidden, n_heads, head_dim)
                # target: n h (n_heads, head_dim)
                
                # This projection only needs to be computed ONCE per layer (weights static).
                # But inside loop is fine for now or we cache it.
                # Let's compute it here.

                # Cast refusal_direction to match model weights (e.g. Float16) if needed
                if refusal_direction.dtype != W_O_reshaped.dtype:
                    refusal_direction = refusal_direction.to(W_O_reshaped.dtype)
                
                effective_dir = torch.einsum('d, dnh -> nh', refusal_direction, W_O_reshaped)
                
                # Score = dot product of head output 'z' with 'effective_dir'
                # z_per_head: [nh, hd]
                # effective_dir: [nh, hd]
                # scores: [nh]
                
                scores = (z_per_head * effective_dir).sum(dim=1) # [n_heads]
                
                total_scores[i] += scores
                
        return total_scores / len(prompts)

    def get_all_heads_attention_breakdown(
        self,
        prompts: List[Prompt],
        group_name: str = "prompts"
    ) -> Dict:
        """
        Compute attention breakdown for ALL heads across ALL layers.
        This is the key function for finding which specific heads show attention hijacking.

        For each head in each layer, computes:
        - % attention to instruction
        - % attention to adversarial suffix (if present)
        - % attention to system/template tokens

        Args:
            prompts: List of prompts to analyze
            group_name: Name for this group (e.g., "refused" or "complied")

        Returns:
            {
                'instruction_attn': [n_layers, n_heads, n_prompts],  # numpy array
                'suffix_attn': [n_layers, n_heads, n_prompts],       # numpy array
                'system_attn': [n_layers, n_heads, n_prompts],       # numpy array
                'user_prefix_attn': [n_layers, n_heads, n_prompts], # numpy array
                'n_layers': int,
                'n_heads': int,
                'n_prompts': int,
                'group_name': str,
                'prompts': List[Prompt]
            }
        """
        print(f"\n{'='*80}")
        print(f"SWEEPING ALL HEADS: {group_name}")
        print(f"{'='*80}")

        n_layers = self.model.config.num_hidden_layers
        n_heads = self.model.config.num_attention_heads
        n_prompts = len(prompts)

        print(f"Model: {n_layers} layers × {n_heads} heads = {n_layers * n_heads} total heads")
        print(f"Prompts: {n_prompts}")
        print(f"Total forward passes: {n_prompts} (all heads analyzed per pass)")

        # Initialize storage: [layers, heads, prompts]
        instruction_attn = np.zeros((n_layers, n_heads, n_prompts))
        suffix_attn = np.zeros((n_layers, n_heads, n_prompts))
        user_prefix_attn = np.zeros((n_layers, n_heads, n_prompts))
        system_attn = np.zeros((n_layers, n_heads, n_prompts))

        # Process each prompt
        for prompt_idx, prompt in enumerate(tqdm(prompts, desc=f"Analyzing {group_name}")):
            pos_info = self.formatter.get_positions(prompt)

            # Tokenize and run model
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

            # outputs.attentions is a tuple of [batch, heads, seq_len, seq_len]
            # We want attention AT t_post (where model starts generating response)
            t_post_idx = pos_info.t_post

            # Process each layer
            for layer_idx in range(n_layers):
                layer_attn = outputs.attentions[layer_idx]  # [1, H, S, S]

                # Get attention from t_post to all tokens: [1, H, S]
                attn_at_post = layer_attn[0, :, t_post_idx, :]  # [H, S]

                # Process each head separately (NO aggregation!)
                for head_idx in range(n_heads):
                    head_attn = attn_at_post[head_idx, :]  # [S]

                    # Calculate attention mass for each region
                    # Region 1: User prefix (template before instruction)
                    user_prefix_mass = head_attn[:pos_info.inst_start].sum().item() if pos_info.inst_start > 0 else 0.0

                    # Region 2: Instruction
                    instr_mass = head_attn[pos_info.inst_start : pos_info.t_inst + 1].sum().item()

                    # Region 3: Adversarial suffix (if present)
                    suffix_mass = 0.0
                    if prompt.has_adv:
                        suffix_mass = head_attn[pos_info.adv_start : pos_info.adv_end + 1].sum().item()

                    # Region 4: System (user_end + assistant_start)
                    system_mass = 1.0 - user_prefix_mass - instr_mass - suffix_mass

                    # Store results
                    instruction_attn[layer_idx, head_idx, prompt_idx] = instr_mass
                    suffix_attn[layer_idx, head_idx, prompt_idx] = suffix_mass
                    user_prefix_attn[layer_idx, head_idx, prompt_idx] = user_prefix_mass
                    system_attn[layer_idx, head_idx, prompt_idx] = system_mass

        print(f"\n✓ Completed sweep: {n_layers} layers × {n_heads} heads × {n_prompts} prompts")

        return {
            'instruction_attn': instruction_attn,
            'suffix_attn': suffix_attn,
            'user_prefix_attn': user_prefix_attn,
            'system_attn': system_attn,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'n_prompts': n_prompts,
            'group_name': group_name,
            'prompts': prompts
        }

    def get_specific_head_attention(
        self,
        prompts: List[Prompt],
        layer_idx: int,
        head_idx: int
    ) -> List[Dict]:
        """
        Get full attention grid for a specific head.
        """
        print(f"\nExtracting Attention Grid for L{layer_idx}H{head_idx}...")
        
        results = []
        
        for prompt in tqdm(prompts):
            pos_info = self.formatter.get_positions(prompt)
            t_post = pos_info.t_post
            
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
                
            # outputs.attentions is tuple of [batch, heads, seq, seq]
            # Get specific layer and head
            # [1, heads, seq, seq] -> [seq, seq]
            attn_grid = outputs.attentions[layer_idx][0, head_idx, :, :].cpu().numpy()
            
            # Slice to relevant part (up to t_post)
            # Both query and key dimensions
            valid_len = t_post + 1
            attn_grid = attn_grid[:valid_len, :valid_len]
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            tokens = tokens[:valid_len]
            
            results.append({
                'tokens': tokens,
                'attention': attn_grid,
                'instruction': prompt.instruction,
                'has_adv': prompt.has_adv,
                'pos_info': {
                    't_inst': pos_info.t_inst,
                    't_post': pos_info.t_post,
                    'adv_start': pos_info.adv_start,
                    'adv_end': pos_info.adv_end
                }
            })
            
        return results
