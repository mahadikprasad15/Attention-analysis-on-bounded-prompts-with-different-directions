from typing import Optional
from src.config import CHAT_TEMPLATES
from src.data.structures import Prompt, TokenPositions

class PromptFormatter:
    """Handles creation and parsing of prompts"""
    
    def __init__(self, tokenizer, template_style: str = "llama3"):
        self.tokenizer = tokenizer
        self.template_style = template_style
        self.template = CHAT_TEMPLATES[template_style]
    
    def create_prompt(self, instruction: str, adv_suffix: Optional[str] = None) -> Prompt:
        """
        Create a formatted prompt
        
        Args:
            instruction: The user instruction (harmful or harmless)
            adv_suffix: Optional adversarial suffix to insert
        
        Returns:
            Prompt object with formatted text and metadata
        """
        # Build prompt text
        text = self.template['user_start']
        text += instruction
        
        # Add adversarial suffix if provided (BEFORE closing user turn)
        if adv_suffix is not None:
            text += " " + adv_suffix
        
        # Close user turn and start assistant turn
        text += self.template['user_end']
        text += self.template['assistant_start']
        
        return Prompt(
            text=text,
            instruction=instruction,
            has_adv=(adv_suffix is not None),
            adv_suffix=adv_suffix,
            template_style=self.template_style
        )
    
    def get_positions(self, prompt: Prompt, debug: bool = False) -> TokenPositions:
        """
        Find token positions for key locations in prompt

        Args:
            prompt: Formatted prompt
            debug: If True, print detailed position information

        Returns:
            TokenPositions with all key indices
        """
        tokens = self.tokenizer.encode(prompt.text, add_special_tokens=False)

        if debug:
            print("\n" + "="*80)
            print("DEBUG: Position Finding")
            print("="*80)
            print(f"Full prompt text:\n{prompt.text}")
            print(f"\nHas adversarial suffix: {prompt.has_adv}")
            if prompt.has_adv:
                print(f"Adversarial suffix: '{prompt.adv_suffix}'")
        
        # Find marker tokens
        user_end_tokens = self.tokenizer.encode(
            self.template['user_end'], add_special_tokens=False
        )
        
        # Find position of user_end marker
        # We search from the end to find the last occurrence
        user_end_pos = None
        
        # Naive substring search
        # Note: In production we might want KMP or similar if performance matters
        # But prompts are short.
        for i in range(len(tokens) - len(user_end_tokens) + 1):
             if tokens[i:i+len(user_end_tokens)] == user_end_tokens:
                user_end_pos = i
                # Keep searching to find the LAST one if there are multiple? 
                # Actually for this template structure, there should only be one user_end before assistant start
                # But let's assume valid structure.
        
        # If we can't find it easily this way (tokenization mismatches), we might need robust alignment
        if user_end_pos is None:
            # Fallback: Approximate by length?
            # Or tokenization drift issue.
            # Let's try to be robust: tokenize parts independently
            # This is safer for Llama-3 specialized tokens
            # But let's raise error for now to catch issues early
            raise ValueError(f"Could not find user_end marker in prompt. Tokens: {tokens[:10]}...")
        
        # Define positions
        # CRITICAL: We need to distinguish between:
        # 1. t_inst_instruction: Last token of the INSTRUCTION ONLY (for harmfulness probe)
        # 2. t_inst_final: Last token before user_end (may include adv suffix)

        # First, find where instruction ends (without suffix)
        prefix_str = self.template['user_start'] + prompt.instruction
        prefix_tokens = self.tokenizer.encode(prefix_str, add_special_tokens=False)

        # The last token of instruction only (before any space or suffix)
        t_inst_instruction = len(prefix_tokens) - 1

        # The last token before user_end marker (includes suffix if present)
        t_inst_final = user_end_pos - 1

        positions = TokenPositions(
            t_inst=t_inst_instruction,  # Use instruction-only position for harmfulness probe
            t_post=len(tokens) - 1,    # Last token of prompt
            total_length=len(tokens)
        )

        # If adversarial suffix is present, mark those positions
        if prompt.has_adv:
            # adv starts right after instruction (usually includes a space token)
            positions.adv_start = len(prefix_tokens)  # First token after instruction
            positions.adv_end = t_inst_final  # Last token before user_end

        if debug:
            print(f"\nToken count: {len(tokens)}")
            print(f"\nTokenized sequence (first 20 tokens):")
            for i, tok_id in enumerate(tokens[:20]):
                tok_str = self.tokenizer.decode([tok_id])
                print(f"  [{i:3d}] {tok_id:6d} → '{tok_str}'")

            if len(tokens) > 20:
                print(f"  ... ({len(tokens) - 20} more tokens)")

            print(f"\nKey positions:")
            print(f"  t_inst (instruction end): {positions.t_inst}")
            print(f"    → Token: '{self.tokenizer.decode([tokens[positions.t_inst]])}'")
            print(f"  t_post (prompt end): {positions.t_post}")
            print(f"    → Token: '{self.tokenizer.decode([tokens[positions.t_post]])}'")

            if prompt.has_adv:
                print(f"  adv_start: {positions.adv_start}")
                print(f"  adv_end: {positions.adv_end}")
                print(f"    → ADV tokens: '{self.tokenizer.decode(tokens[positions.adv_start:positions.adv_end+1])}'")
                print(f"    → Expected: '{prompt.adv_suffix}'")

            print("="*80 + "\n")

        return positions
