import os
import pandas as pd
from typing import List

DATA_DIR = "data"

def load_harmful_instructions() -> List[str]:
    """Load harmful instructions from AdvBench CSV"""
    path = os.path.join(DATA_DIR, "harmful_behaviors.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}. Please run scripts/download_data.py")
    
    df = pd.read_csv(path)
    # Column is usually 'goal'
    return df['goal'].tolist()

def load_harmless_instructions() -> List[str]:
    """Load harmless instructions from processed Alpaca CSV"""
    path = os.path.join(DATA_DIR, "harmless_alpaca.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}. Please run scripts/download_data.py")
    
    df = pd.read_csv(path)
    return df['goal'].tolist()
