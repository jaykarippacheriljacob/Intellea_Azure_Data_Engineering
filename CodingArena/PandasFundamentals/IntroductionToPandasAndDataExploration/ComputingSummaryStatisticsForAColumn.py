"""
    Computing Summary Statistics for a Column
    Medium
    Introduction to Pandas and Data Exploration
    Write summary_stats(records, column). Build a DataFrame and return
    {"min": ..., "max": ..., "mean": ...} for the numeric column column,
    with mean rounded to 4 decimals.

    Examples

    Arguments: [{'marks': 70}, {'marks': 80}, {'marks': 90}], 'marks'
    Returns: {'min': 70.0, 'max': 90.0, 'mean': 80.0}
    Explanation: min 70.0, max 90.0, mean 80.0.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds
    numeric values. 1 <= len(records) <= 1000
"""
import pandas as pd      # REVISION 2: the import was written twice; once is enough


def summary_stats(records, column):
    """
    Return {"min", "max", "mean"} of `column` as plain Python floats,
    with mean rounded to 4 decimals.

    Time:  O(n) - one pass each for min, max and mean.
    Space: O(n) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame, then take the one column (a Series).
    df = pd.DataFrame(records)
    col = df[column]

    # REVISION 1 (bug fix): the first version returned the mean unrounded.
    # The example passed only because 80.0 has no decimals; [70, 80, 85]
    # gives 78.33333333333333 instead of the expected 78.3333.
    #
    # REVISION 3: float(). pandas returns NumPy scalars, not Python numbers:
    #     col.min()             -> np.int64(70)       (expected 70.0)
    #     round(col.mean(), 4)  -> np.float64(78.3333)
    # float() gives plain Python floats, matching the 70.0 / 90.0 format.
    # Order: compute -> float() -> round(). Never round the inputs.
    return {
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": round(float(col.mean()), 4),
    }


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas):
#        vals = [r[column] for r in records]
#        return {"min": float(min(vals)), "max": float(max(vals)),
#                "mean": round(sum(vals) / len(vals), 4)}
#
# 2. pandas one-call version:
#        s = df[column].agg(["min", "max", "mean"])
#        return {"min": float(s["min"]), "max": float(s["max"]),
#                "mean": round(float(s["mean"]), 4)}
#
# 3. Why the grader message said "check the return value / types":
#      - printing is not returning; the grader reads the return value
#      - an unrounded mean fails as soon as it has many decimals
#      - NumPy scalars (np.int64, np.float64) may not match the expected
#        plain floats in a strict check
#
# 4. Python's round() uses "banker's rounding" on exact halves:
#        round(0.125, 2) -> 0.12, round(2.5) -> 2
#    It rarely matters with 4 decimals, but is a known interview trap.
#
# 5. Edge cases worth naming in an interview:
#      - single row          -> min = max = mean
#      - all equal values    -> min = max = mean
#      - negative values     -> work as normal
#      - floats in the data  -> work as normal
#      - NaN in the column   -> pandas skips NaN in min/max/mean
#                               (Python's min/max would not)
#
# 6. Related: df.describe() gives count, mean, std, min, quartiles, max
#    in one call (Pandas Fundamentals 1.7).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'v': i} for i in range(1, 1001)]

    tests = [
        # (records, column, expected, description)
        ([{'marks': 70}, {'marks': 80}, {'marks': 90}], 'marks',
         {'min': 70.0, 'max': 90.0, 'mean': 80.0}, "given example"),
        ([{'marks': 70}, {'marks': 80}, {'marks': 85}], 'marks',
         {'min': 70.0, 'max': 85.0, 'mean': 78.3333}, "mean needs rounding"),
        ([{'x': 1}, {'x': 2}, {'x': 2}], 'x',
         {'min': 1.0, 'max': 2.0, 'mean': 1.6667}, "rounds up at 4th decimal"),
        ([{'x': 42}], 'x',
         {'min': 42.0, 'max': 42.0, 'mean': 42.0}, "single row"),
        ([{'x': -5}, {'x': -1}, {'x': 3}], 'x',
         {'min': -5.0, 'max': 3.0, 'mean': -1.0}, "negative values"),
        ([{'p': 1.5}, {'p': 2.25}], 'p',
         {'min': 1.5, 'max': 2.25, 'mean': 1.875}, "float values"),
        ([{'a': 1, 'b': 10}, {'a': 2, 'b': 20}], 'b',
         {'min': 10.0, 'max': 20.0, 'mean': 15.0}, "other columns ignored"),
        (big, 'v', {'min': 1.0, 'max': 1000.0, 'mean': 500.5}, "max size n=1000"),
    ]

    passed = 0
    for records, column, expected, desc in tests:
        got = summary_stats(records, column)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<26} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = summary_stats([{'m': 70}, {'m': 80}, {'m': 85}], 'm')
    assert all(type(v) is float for v in out.values()), "plain Python floats"
    assert list(out) == ["min", "max", "mean"], "key order"
    print("PASS: plain float types / key order")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()