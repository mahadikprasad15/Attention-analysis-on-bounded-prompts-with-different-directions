import os
import requests
import json
import pandas as pd
from typing import List

DATA_DIR = "data"

def download_file(url: str, dest_path: str):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    print("✓ Download complete.")

def process_advbench():
    """Download and process AdvBench harmful behaviors"""
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    dest = os.path.join(DATA_DIR, "harmful_behaviors.csv")
    download_file(url, dest)
    
    # Verify we can read it
    df = pd.read_csv(dest)
    print(f"✓ Loaded AdvBench: {len(df)} examples")
    return df

def process_alpaca():
    """Download and process Alpaca (harmless instructions)"""
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    dest = os.path.join(DATA_DIR, "alpaca_data.json")
    download_file(url, dest)
    
    with open(dest, 'r') as f:
        data = json.load(f)
    
    # Filter for standard instructions (no input) to match AdvBench format
    # AdvBench is just "goal". Alpaca is "instruction" + optional "input".
    # We want simple instructions like "How do I..."
    
    clean_instructions = []
    for item in data:
        if not item.get('input'):  # No context input
            clean_instructions.append(item['instruction'])
            
    # Save as CSV for consistency
    df = pd.DataFrame({'goal': clean_instructions})
    out_path = os.path.join(DATA_DIR, "harmless_alpaca.csv")
    df.to_csv(out_path, index=False)
    print(f"✓ Processed Alpaca: {len(df)} examples (saved to {out_path})")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print("Step 1: Fetching Harmful Data (AdvBench)...")
    process_advbench()
    
    print("\nStep 2: Fetching Harmless Data (Alpaca)...")
    process_alpaca()
    
    print("\n✓ Data setup complete!")

if __name__ == "__main__":
    main()
