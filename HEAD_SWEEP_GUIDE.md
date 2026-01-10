# Head-Level Sweep Analysis: Finding Attention Hijacking Heads

## Overview

This feature addresses a critical limitation: **aggregating attention across all heads may hide the specific heads responsible for attention hijacking**.

The paper "Universal jailbreak suffixes are strong attention hijackers" suggests that adversarial suffixes capture significant attention. However, aggregated results might show only 3-5% suffix attention because:

1. **Most heads might ignore the suffix** (~90% of heads)
2. **A few specialized heads might show 50-70% suffix attention**
3. **Averaging hides this pattern**

## What This Does

The head-level sweep analyzes **ALL 704 heads** (22 layers × 32 heads) separately to:

1. **Identify hijacking heads** - Which specific heads show high suffix attention?
2. **Compare groups** - How do heads differ between refused (clean) vs complied (jailbreak)?
3. **Layer analysis** - Which layers contain the most hijacking heads?
4. **Mechanism insight** - Is hijacking concentrated or distributed?

## Computational Cost

**Very cheap!** The expensive part is the forward pass with `output_attentions=True`, which we're already doing.

- **Forward passes needed**: Same as before (once per prompt)
- **Additional processing**: Pure tensor indexing and summation
- **Memory**: Reuses already-extracted attention tensors
- **Time overhead**: ~10 seconds for 50 prompts

## New Visualizations Generated

When you run `python main.py --n_samples 50`, you'll get these new visualizations:

### 1. **Heatmaps** (layers × heads)
- `heads_suffix_heatmap_complied.png` - Suffix attention for jailbreak prompts
- `heads_suffix_heatmap_refused.png` - Suffix attention for clean prompts
- `heads_instruction_heatmap_complied.png` - Instruction attention for jailbreak
- `heads_instruction_heatmap_refused.png` - Instruction attention for clean

**What to look for**:
- **Hot spots** (red regions) = heads with high attention to that region
- **Sparse patterns** = hijacking concentrated in few heads
- **Layer patterns** = which layers contain hijacking heads?

### 2. **Top Heads Bar Chart**
- `top20_suffix_heads_complied.png` - Top 20 heads by suffix attention

**What to look for**:
- **Top head value** - If >30%, that's a strong hijacking head!
- **Drop-off pattern** - Steep drop = few specialized heads; gradual = distributed
- **Layer distribution** - Are top heads from one layer or multiple?

### 3. **Layer-wise Summary**
- `layer_summary_refused.png` - Attention distribution per layer (refused)
- `layer_summary_complied.png` - Attention distribution per layer (complied)

**What to look for**:
- **Which layers** show highest suffix attention?
- **Early vs late** layers - hijacking might occur in middle layers
- **Comparison** - how does complied differ from refused?

### 4. **Comparison Heatmaps** (side-by-side)
- `comparison_suffix_heads.png` - Refused vs Complied (suffix)
- `comparison_instruction_heads.png` - Refused vs Complied (instruction)

**What to look for**:
- **Difference pattern** - which heads change most between groups?
- **Shared attention** - do some heads always attend to system/template?
- **Hijacking signature** - specific heads that only light up with suffix

## Console Output

When the sweep completes, you'll see:

```
================================================================================
HEAD SWEEP SUMMARY
================================================================================

Top Suffix Attention Head:
  Layer 15, Head 8
  Suffix Attention: 42.3%

Top 5 Heads by Suffix Attention:
  1. L15H8: 42.3%
  2. L14H12: 38.7%
  3. L16H3: 35.2%
  4. L15H21: 31.8%
  5. L13H9: 28.4%

Layer with Highest Average Suffix Attention:
  Layer 15: 18.7%

================================================================================
```

## How to Use

### Standard Run
```bash
python main.py --n_samples 50
```

This automatically runs the head sweep if there are both refused and complied samples.

### What to Look For

**Scenario 1: Strong Hijacking Pattern**
- Top head shows >30% suffix attention
- Top 5 heads all from layers 12-16
- Clear hot spots in heatmap
- **Interpretation**: Attention hijacking is real and concentrated

**Scenario 2: Weak Hijacking Pattern**
- Top head shows <10% suffix attention
- Top heads distributed across all layers
- Heatmap shows no clear hot spots
- **Interpretation**: Hijacking might not be the mechanism, or:
  - Suffixes don't transfer well to this model size
  - Hijacking occurs via residual stream, not attention
  - Need to check earlier layers

**Scenario 3: Layer-Specific Pattern**
- All top heads from layers 10-15
- Early/late layers show low suffix attention
- **Interpretation**: Hijacking is layer-specific, might intervene there

## Expected Results

Based on the paper, we expect:

### For Jailbreak Complied Prompts:
- **Top heads**: 20-40% suffix attention
- **Location**: Middle-to-late layers (10-18)
- **Count**: 5-20 specialized heads
- **Pattern**: Clear hot spots in heatmap

### For Clean Refused Prompts:
- **Suffix attention**: 0% (no suffix present)
- **Instruction attention**: 40-60%
- **Pattern**: Different heads activate compared to complied

### Key Comparison:
- **Hijacking heads** should show:
  - HIGH suffix attention in complied group
  - HIGH instruction attention in refused group
  - Interpretation: These heads attend to "user content" and get hijacked by suffix

## Interpreting Low Aggregate Attention

If you previously saw **3% aggregate suffix attention**, the head sweep might reveal:

```
Aggregate (mean across all heads): 3.2%

But head-level analysis shows:
  - Head L15H8: 45% suffix attention
  - Head L14H12: 38% suffix attention
  - Head L16H3: 32% suffix attention
  - ... (3 heads above 30%)
  - ... (29 heads with <1%)

Average = (45 + 38 + 32 + ... + 29×<1%) / 32 heads = 3.2%
```

**The low aggregate hides the high specialization!**

## Troubleshooting

### "No complied samples found"
- Your suffixes might not be working
- Check baseline evaluation: does condition J have lower refusal rate than R?
- Try different suffixes or optimize new ones for your model

### "All heads show low suffix attention"
- Hijacking might occur in earlier layers - check layer summary
- Try analyzing layers 5-15 instead of just last layer
- Mechanism might not be attention-based (check residual stream)

### "Heatmap is all yellow/orange, no clear pattern"
- Increase color scale sensitivity
- Use difference heatmap (complied - refused) to see changes
- Look at top heads bar chart instead

## Files Modified

- `src/analysis/attention_pattern.py`: Added `get_all_heads_attention_breakdown()`
- `src/analysis/visualization.py`: Added 5 new visualization functions
- `main.py`: Integrated sweep analysis into pipeline

## Next Steps After Running

1. **Identify top hijacking heads** from console output
2. **Examine heatmaps** to find patterns
3. **Compare with refusal contribution heads** (already computed)
4. **Test intervention** on specific heads
5. **Analyze token-level attention** for top heads

## Research Questions This Answers

✅ **Which specific heads show attention hijacking?**
   → Top N heads bar chart

✅ **Is hijacking concentrated or distributed?**
   → Heatmap patterns (sparse vs dense)

✅ **Which layers are responsible?**
   → Layer summary and top heads layer distribution

✅ **Do hijacking heads also contribute to refusal?**
   → Cross-reference with head contribution analysis

✅ **Why is aggregate attention low despite hijacking?**
   → Compare individual head values to aggregate

## Performance

For Llama-3.2-1B (22 layers, 32 heads):
- **Total heads analyzed**: 704
- **Time per prompt**: ~same as before (single forward pass)
- **Total overhead**: ~10 seconds for 50 prompts
- **Memory**: Minimal (reuses existing tensors)

## Citation

If this analysis reveals interesting patterns, consider mentioning:

> We performed head-level analysis across all 704 attention heads to identify
> specific heads responsible for attention hijacking, revealing that [your findings].

---

**Ready to find your hijacking heads!** 🎯

Run with:
```bash
python main.py --force_regen --n_samples 50
```
