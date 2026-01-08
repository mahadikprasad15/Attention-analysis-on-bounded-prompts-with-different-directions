from dataclasses import dataclass
from typing import Optional

@dataclass
class Prompt:
    """Represents a formatted prompt with metadata"""
    text: str
    instruction: str
    has_adv: bool
    adv_suffix: Optional[str]
    template_style: str
    
    def __str__(self):
        return self.text

@dataclass
class TokenPositions:
    """Token positions in a prompt"""
    t_inst: int          # End of instruction
    t_post: int          # End of template (before generation)
    adv_start: Optional[int] = None  # Start of adversarial suffix
    adv_end: Optional[int] = None    # End of adversarial suffix
    total_length: int = 0
    
    def has_adversarial(self) -> bool:
        return self.adv_start is not None
