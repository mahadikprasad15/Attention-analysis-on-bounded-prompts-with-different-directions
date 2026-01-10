# Bug Fixes: Attention Analysis on Adversarial Suffixes

## Issues Identified

### Issue 1: Incorrect Attention Region Boundaries (CRITICAL)

**Problem:**
The instruction attention mass calculation was including template prefix tokens (like `<|start_header_id|>user<|end_header_id|>`), artificially inflating the "instruction" attention to ~72%.

**Root Cause:**
In `src/analysis/attention_pattern.py:82`, the code calculated:
```python
instr_mass = avg_attn[:pos_info.t_inst + 1].sum().item()
```

This summed attention from token 0 to t_inst, which included:
- Template prefix (`<|start_header_id|>user<|end_header_id|>\n\n`) - several tokens
- Actual instruction text

**Impact:**
- Inflated instruction attention (~72%)
- Deflated suffix attention (~3.5%)
- Made it appear that suffixes were barely attended to
- Impossible to distinguish template attention from instruction attention

**Fix:**
1. Added `inst_start` field to `TokenPositions` dataclass to track where instruction begins (after template prefix)
2. Updated `get_positions()` in `src/data/formatting.py` to calculate `inst_start`
3. Updated attention mass calculation to use correct boundaries:
   ```python
   user_prefix_mass = avg_attn[:pos_info.inst_start].sum().item()  # Template
   instr_mass = avg_attn[pos_info.inst_start : pos_info.t_inst + 1].sum().item()  # Instruction only
   ```
4. Now tracking 4 categories: `instruction_attn`, `suffix_attn`, `user_prefix_attn`, `system_attn`

---

### Issue 2: Comparing Wrong Datasets (CRITICAL)

**Problem:**
The comparison between "refused" and "complied" samples was comparing two subgroups from the SAME dataset (J - jailbreak with suffixes):
- Refused: J prompts that still refused despite having suffix
- Complied: J prompts that complied because of suffix

**Root Cause:**
In `main.py:379-380`:
```python
refused_samples = [res for res in j_intervened if res.actually_refuses]
complied_samples = [res for res in j_intervened if not res.actually_refuses]
```

Both groups had adversarial suffixes!

**Impact:**
- Attention patterns between groups were almost identical (~69% vs ~74% instruction, ~4% vs ~3% suffix)
- No meaningful insight into how suffixes change attention
- Couldn't determine if low suffix attention was real or a measurement artifact

**Fix:**
Changed comparison to compare the RIGHT datasets:
- **Refused baseline**: Clean prompts from dataset R (NO suffix) that refused
- **Jailbreak complied**: Prompts from dataset J (WITH suffix) that complied

This now shows the actual difference in attention patterns between:
- Normal refusal behavior (clean prompts)
- Successful jailbreak (prompts with adversarial suffix)

---

### Issue 3: Visualization Code Not Updated

**Problem:**
Visualization functions expected 3 categories but now we have 4.

**Fix:**
Updated all visualization functions to:
1. Handle the new `user_prefix_attn` field
2. Combine `user_prefix_attn` and `system_attn` into "System/Template" for cleaner visualizations
3. Updated labels to reflect "Clean Refused (No Suffix)" vs "Jailbreak Complied (With Suffix)"

## Files Modified

1. `src/data/structures.py` - Added `inst_start` field to `TokenPositions`
2. `src/data/formatting.py` - Calculate `inst_start` in `get_positions()`
3. `src/analysis/attention_pattern.py` - Fixed attention mass calculation with correct boundaries
4. `src/analysis/visualization.py` - Updated all visualization functions
5. `main.py` - Fixed dataset comparison logic
6. `test_attention_fixes.py` - Created test suite to verify fixes

## Expected Results After Fix

### Before (Incorrect):
- **Jailbreak Condition Overall:**
  - Instruction: 72.79% (inflated - includes template)
  - Adv Suffix: 3.52% (may be accurate but unclear)
  - System/Other: 23.69%

- **Refused vs Complied (both from J):**
  - Almost identical patterns (both ~69-74% instruction, ~3-4% suffix)
  - No meaningful difference visible

### After (Correct):
- **Jailbreak Condition Overall:**
  - Instruction: ~40-50% (actual instruction text only)
  - Adv Suffix: ~3-10% (clearer signal)
  - User Prefix: ~10-15% (template tokens)
  - System/Other: ~20-30% (assistant start, etc.)

- **Clean Refused vs Jailbreak Complied:**
  - Clean refused (no suffix): 0% suffix attention (no suffix present)
  - Jailbreak complied (with suffix): X% suffix attention
  - Clear difference in attention patterns visible

## Key Insights Enabled by Fixes

1. **Separate template from instruction**: Now can see how much attention goes to the actual instruction vs the template formatting
2. **Meaningful comparisons**: Compare normal refusal (no suffix) vs jailbreak success (with suffix)
3. **True suffix attention**: With clean baseline, can determine if suffix attention is genuinely low or if there's still an issue

## Recommendations for Next Steps

1. **Re-run the full analysis** with `--force_regen` to regenerate all datasets and results with the fixes:
   ```bash
   python main.py --force_regen --n_samples 50
   ```

2. **Examine new visualizations** to see:
   - What percentage of attention actually goes to the suffix in successful jailbreaks
   - How attention patterns differ between clean refused and jailbreak complied
   - Whether suffix attention is genuinely low (interesting finding) or there's still a measurement issue

3. **If suffix attention is still very low (<5%)**, investigate:
   - Are suffixes working through **indirect** mechanisms rather than direct attention?
   - Do they affect attention in **earlier layers** rather than the last layer?
   - Do they modify the **residual stream** without much direct attention?
   - Try analyzing attention at different layers: `layer_idx=15` or `layer_idx=20`

4. **Debug token positions** with a real example:
   ```python
   from src.data.formatting import PromptFormatter
   from src.config import Config

   config = Config()
   formatter = PromptFormatter(tokenizer, template_style='llama3')
   prompt = formatter.create_prompt("harmful instruction", adv_suffix=config.adv_suffixes[0])
   positions = formatter.get_positions(prompt, debug=True)  # Set debug=True to see full breakdown
   ```

## Testing

Run the test suite to verify all fixes work correctly:
```bash
python test_attention_fixes.py
```

All tests should pass:
- ✓ TEST 1: Token Position Structure
- ✓ TEST 2: Attention Region Boundaries
- ✓ TEST 3: Dataset Comparison Logic

## Questions to Investigate After Re-running

1. **What is the actual suffix attention percentage** in successful jailbreaks?
2. **How does it compare to instruction attention** in the same prompts?
3. **Is there a pattern** - do certain heads attend more to the suffix?
4. **Does suffix attention correlate** with jailbreak success rate?
5. **Are there layer-specific effects** - does suffix attention peak in certain layers?

---

**Conclusion**: These fixes address fundamental measurement and comparison errors in the analysis. The results should now accurately reflect where the model is attending and enable meaningful investigation of how adversarial suffixes work mechanistically.
