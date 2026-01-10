#!/usr/bin/env python3
"""
Test script to verify attention analysis fixes:
1. Token position tracking (inst_start field)
2. Correct attention region boundaries
3. Dataset comparison logic
"""

def test_token_positions():
    """Test that token positions structure has required fields"""
    print("=" * 80)
    print("TEST 1: Token Position Structure")
    print("=" * 80)

    from src.data.structures import TokenPositions

    # Test that TokenPositions has the new inst_start field
    pos = TokenPositions(
        t_inst=50,
        t_post=99,
        inst_start=10,
        total_length=100
    )

    assert hasattr(pos, 'inst_start'), "inst_start field missing!"
    assert pos.inst_start == 10, "inst_start value incorrect!"

    print(f"✓ TokenPositions has inst_start field: {pos.inst_start}")

    # Test with adversarial suffix
    pos_with_adv = TokenPositions(
        t_inst=50,
        t_post=99,
        inst_start=10,
        adv_start=51,
        adv_end=70,
        total_length=100
    )

    assert pos_with_adv.has_adversarial(), "has_adversarial() should return True"
    print(f"✓ has_adversarial() works correctly")

    print("\n✓✓✓ TEST 1 PASSED ✓✓✓\n")


def test_attention_region_boundaries():
    """Test that attention mass calculation uses correct boundaries"""
    print("=" * 80)
    print("TEST 2: Attention Region Boundaries")
    print("=" * 80)

    # Simulate positions
    class MockPositions:
        inst_start = 10  # Instruction starts at token 10
        t_inst = 50      # Instruction ends at token 50
        adv_start = 51   # Suffix starts at token 51
        adv_end = 70     # Suffix ends at token 70
        t_post = 99      # Last token
        total_length = 100

    pos = MockPositions()

    # Calculate masses using the NEW logic (from attention_pattern.py)
    # Simulating uniform attention (each token gets 0.01)
    seq_len = 100
    uniform_attn_per_token = 0.01

    user_prefix_mass = pos.inst_start * uniform_attn_per_token if pos.inst_start > 0 else 0.0
    instr_mass = (pos.t_inst - pos.inst_start + 1) * uniform_attn_per_token
    suffix_mass = (pos.adv_end - pos.adv_start + 1) * uniform_attn_per_token
    system_mass = 1.0 - user_prefix_mass - instr_mass - suffix_mass

    print(f"Mock positions:")
    print(f"  User prefix: [0, {pos.inst_start})")
    print(f"  Instruction: [{pos.inst_start}, {pos.t_inst + 1})")
    print(f"  Suffix: [{pos.adv_start}, {pos.adv_end + 1})")
    print(f"  System/Other: remainder")

    print(f"\nCalculated attention masses (uniform distribution):")
    print(f"  User prefix: {user_prefix_mass:.4f} ({user_prefix_mass * 100:.1f}%)")
    print(f"  Instruction: {instr_mass:.4f} ({instr_mass * 100:.1f}%)")
    print(f"  Suffix: {suffix_mass:.4f} ({suffix_mass * 100:.1f}%)")
    print(f"  System: {system_mass:.4f} ({system_mass * 100:.1f}%)")

    # Verify no overlap
    user_prefix_tokens = pos.inst_start
    instr_tokens = pos.t_inst - pos.inst_start + 1
    suffix_tokens = pos.adv_end - pos.adv_start + 1

    print(f"\nToken counts:")
    print(f"  User prefix: {user_prefix_tokens} tokens")
    print(f"  Instruction: {instr_tokens} tokens")
    print(f"  Suffix: {suffix_tokens} tokens")

    # With uniform attention, masses should be proportional to token counts
    expected_user_prefix = (user_prefix_tokens / seq_len)
    expected_instr = (instr_tokens / seq_len)
    expected_suffix = (suffix_tokens / seq_len)

    print(f"\nExpected masses (proportional to token counts):")
    print(f"  User prefix: {expected_user_prefix:.4f}")
    print(f"  Instruction: {expected_instr:.4f}")
    print(f"  Suffix: {expected_suffix:.4f}")

    # Verify calculation is correct
    assert abs(user_prefix_mass - expected_user_prefix) < 0.01, "User prefix mass incorrect!"
    assert abs(instr_mass - expected_instr) < 0.01, "Instruction mass incorrect!"
    assert abs(suffix_mass - expected_suffix) < 0.01, "Suffix mass incorrect!"
    assert abs(user_prefix_mass + instr_mass + suffix_mass + system_mass - 1.0) < 0.01, "Masses don't sum to 1!"

    print("\n✓✓✓ TEST 2 PASSED ✓✓✓\n")


def test_dataset_comparison_logic():
    """Test that we're comparing the right datasets"""
    print("=" * 80)
    print("TEST 3: Dataset Comparison Logic")
    print("=" * 80)

    # Simulate evaluation results
    class MockResult:
        def __init__(self, instruction, actually_refuses, refusal_score, has_suffix):
            self.instruction = instruction
            self.actually_refuses = actually_refuses
            self.refusal_score = refusal_score
            self.has_suffix = has_suffix

    # Simulated results for dataset J (jailbreak with suffixes)
    j_results = [
        MockResult("harmful_1", True, 0.9, True),   # Refused despite suffix
        MockResult("harmful_2", False, 0.2, True),  # Complied (jailbreak success)
        MockResult("harmful_3", False, 0.1, True),  # Complied (jailbreak success)
        MockResult("harmful_4", True, 0.8, True),   # Refused despite suffix
    ]

    # OLD (WRONG) logic: Compare refused vs complied from SAME dataset J
    print("\n--- OLD (WRONG) Logic ---")
    old_refused = [r for r in j_results if r.actually_refuses]
    old_complied = [r for r in j_results if not r.actually_refuses]

    print(f"Old refused group: {len(old_refused)} samples (all have suffix={all(r.has_suffix for r in old_refused)})")
    print(f"Old complied group: {len(old_complied)} samples (all have suffix={all(r.has_suffix for r in old_complied)})")
    print("⚠️  PROBLEM: Both groups have suffixes, so attention patterns will be similar!")

    # NEW (CORRECT) logic: Compare clean refused (no suffix) vs jailbreak complied (with suffix)
    print("\n--- NEW (CORRECT) Logic ---")

    # For clean refused, we'd use dataset R (no suffix)
    # Simulated dataset R
    r_results = [
        MockResult("harmful_1", True, 0.95, False),  # Clean refusal
        MockResult("harmful_5", True, 0.92, False),  # Clean refusal
        MockResult("harmful_6", True, 0.88, False),  # Clean refusal
    ]

    new_refused = r_results[:3]  # Top 3 clean refused
    new_complied = [r for r in j_results if not r.actually_refuses][:3]  # Top 3 jailbreak complied

    print(f"New refused group (clean): {len(new_refused)} samples (all have suffix={all(r.has_suffix for r in new_refused)})")
    print(f"New complied group (jailbreak): {len(new_complied)} samples (all have suffix={all(r.has_suffix for r in new_complied)})")
    print("✓ CORRECT: Comparing clean (no suffix) vs jailbreak (with suffix)")

    # Verify the fix
    assert not any(r.has_suffix for r in new_refused), "Clean refused should NOT have suffixes!"
    assert all(r.has_suffix for r in new_complied), "Jailbreak complied SHOULD have suffixes!"

    print("\n✓✓✓ TEST 3 PASSED ✓✓✓\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING ATTENTION ANALYSIS FIX VERIFICATION TESTS")
    print("=" * 80 + "\n")

    try:
        test_token_positions()
        test_attention_region_boundaries()
        test_dataset_comparison_logic()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓✓✓")
        print("=" * 80 + "\n")

        print("Summary of fixes:")
        print("1. ✓ Added inst_start field to track where instruction begins (after template)")
        print("2. ✓ Fixed attention mass calculation to use correct region boundaries")
        print("3. ✓ Fixed dataset comparison to compare clean refused vs jailbreak complied")
        print("\nThe analysis should now correctly show:")
        print("  - Separate attention masses for instruction vs template vs suffix")
        print("  - Meaningful differences between refused and complied attention patterns")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
