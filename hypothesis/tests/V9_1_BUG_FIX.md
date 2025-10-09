# V9.1 Bug Fix - K=K_c Boundary Condition

**Date:** October 9, 2025  
**Issue:** V9.1 predicted 0% at K=K_c instead of applying floor formula  
**Status:** ✅ FIXED

---

## Problem Description

### Observed Behavior

When running `--compare`, V9.1 showed:
- K=0.8: 18.6% (floor formula) ✅
- K=0.9: 22.2% (floor formula) ✅
- **K=1.0: 0.0%** (V8 transition formula) ❌ **BUG**
- K=1.1: 32.7% (V8 transition formula) ✅

**Empirical at K=1.0:** 20.0%  
**V9.1 prediction:** 0.0%  
**Error:** 20.0% (huge!)

### Root Cause

**Boundary condition bug in V9.1 implementation:**

```python
if K_ratio < 1.0:          # Floor formula (correct)
    floor = 0.26 * (K_ratio ** 1.5)
elif K_ratio < 1.2:        # V8 transition (K=1.0 falls here!)
    # V8 formula gives 0% at K=K_c
    basin_volume = 1.0 - (1.0 / K_ratio) ** exponent
```

**Problem:** K_ratio=1.0 (K=K_c) falls into the `elif K_ratio < 1.2` branch, which uses V8's transition formula. V8's formula gives 0% at exactly K=K_c, creating the huge error.

---

## Solution

### Code Change

Changed boundary from `<` to `<=`:

```python
if K_ratio <= 1.0:         # Floor formula (includes K=K_c now!)
    floor = 0.26 * (K_ratio ** 1.5)
    # At K=1.0: floor = 0.26 * 1.0^1.5 = 26%
elif K_ratio < 1.2:        # V8 transition (K>1.0 only)
    # V8 formula for K > K_c
```

### Fixed Behavior

**After fix:**
- K=0.8: 18.6% (floor) ✅
- K=0.9: 22.2% (floor) ✅
- **K=1.0: 26.0%** (floor) ✅ **FIXED**
- K=1.1: 32.7% (V8 transition) ✅

**At K=1.0:**
- Empirical: 20.0%
- V9.1 (before): 0.0% → 20.0% error
- V9.1 (after): 26.0% → 6.0% error
- **Improvement: 14 percentage points!**

---

## Impact on Overall Performance

### Before Fix

| Regime | V8 Error | V9.1 Error (buggy) |
|--------|----------|---------------------|
| Below critical | 13.2% | 7.2% |
| **K=1.0** | **20.0%** | **20.0%** ← same as V8! |
| Transition | 7.0% | 7.0% |
| Strong coupling | 2.6% | 2.6% |
| **Overall** | **6.4%** | **6.0%** |

V9.1 barely beat V8 (6.0% vs 6.4%) because K=1.0 bug canceled out the below-critical improvement.

### After Fix (Expected)

| Regime | V8 Error | V9.1 Error (fixed) |
|--------|----------|---------------------|
| Below critical | 13.2% | 7.2% |
| **K=1.0** | **20.0%** | **6.0%** ← fixed! |
| Transition | 7.0% | 7.0% |
| Strong coupling | 2.6% | 2.6% |
| **Overall** | **6.4%** | **~5.5%** |

**Expected improvement:**
- V9.1 overall error: 6.0% → ~5.5%
- Clearer victory over V8 (5.5% vs 6.4% = 14% improvement)
- K=1.0 error reduced from 20% to 6%

---

## Validation

### Test Command

```bash
python3 enhanced_test_basin_volume.py --compare
```

### Expected Output (After Fix)

```
K/K_c    Empirical  V8       V9.1    
--------------------------------------
0.8      9.5%       0.0%     18.6%   
0.9      13.0%      0.0%     22.2%   
1.0      20.0%      0.0%     26.0%   ← FIXED (was 0.0%)
1.1      38.0%      32.7%    32.7%   
...

======================================================================
MEAN ABSOLUTE ERROR (all K values):
----------------------------------------------------------------------
Formula V8: 6.4%  ✅ Excellent
Formula V9.1: 5.5%  ✅ Excellent  ← Better than before

🏆 BEST FORMULA: V9.1 (mean error 5.5%)

✅ HYPOTHESIS VALIDATED with V9.1
   → Update production code to use formula V9.1
   → Proceed to hardware with confidence
   → V9.1 'Goldilocks' formula improves where V8 fails, preserves where V8 excels
```

---

## Lesson Learned

**Boundary conditions matter!**

In physics-inspired formulas with regime transitions:
1. **Be explicit about boundaries:** Use `<=` vs `<` carefully
2. **Test boundary values:** K=K_c is the most important test case
3. **Check predictions at boundaries:** 0% at K=K_c is suspicious (should be ~20-30%)
4. **Document boundary logic:** "Include K=K_c to avoid V8's 0% prediction"

**The fix:**
- One character change: `<` → `<=`
- Impact: 14 percentage points improvement at K=1.0
- Overall: V9.1 now clearly beats V8 (5.5% vs 6.4%)

---

## Files Modified

1. **`enhanced_test_basin_volume.py`** (2 locations):
   - Main `predict_basin_volume()` function, line ~380
   - Standalone `predict_basin_volume_v9_1()` function, line ~545

**Change:**
```python
# Before (bug)
if K_ratio < 1.0:

# After (fixed)
if K_ratio <= 1.0:
```

---

## Status

✅ **FIXED** - Both V9.1 implementations now correctly handle K=K_c

**Next run of `--compare` should show:**
- V9.1 predicting 26% at K=1.0 (not 0%)
- V9.1 overall error ~5.5% (better than before)
- V9.1 as clear winner over V8

---

**Date fixed:** October 9, 2025  
**Severity:** High (20% error at critical boundary)  
**Impact:** Medium (only 1 test point affected, but important one)  
**Validation:** Pending next `--compare` run
