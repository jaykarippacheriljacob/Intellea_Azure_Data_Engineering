"""
    Cleaning Then Summarizing a Dataset in One Pipeline
    Easy
    End-to-End Pandas Data Processing Project
    Write clean_and_summarize(records, column). Build a DataFrame, drop rows
    where column is missing, and return {"count": ..., "mean": ...} for what
    remains. If NOTHING remains after cleaning, return
    {"count": 0, "mean": None} instead of crashing.

    Examples

    Arguments: [{'marks': 80}, {'marks': None}, {'marks': 90}], 'marks'
    Returns: {'count': 2, 'mean': 85.0}
    Explanation: Only 2 of the 3 rows have a real value; their mean is 85.0.
    Constraints

    records is a non-empty list of dicts sharing the same keys, where column
    may be None. 1 <= len(records) <= 1000
"""
import pandas as pd


def clean_and_summarize(records, column):
    """
    dropna(column) -> empty check -> {"count": int, "mean": float}.

    Time:  O(n) - one pass to drop missing values, one for count/mean.
    Space: O(n) - for the DataFrame.
    """
    # Step 0: list of dicts -> DataFrame. None -> NaN.
    df = pd.DataFrame(records)

    # Step 1: drop rows missing THIS column only.
    # FIRST ATTEMPT (too strict): df.dropna() checks EVERY column.
    s = df.dropna(subset=[column])                       # CHANGED: subset

    # Step 2: nothing left -> the special answer from the task.
    # Without this, mean() on an empty Series gives NaN, not None.
    if s[column].empty:
        return {"count": 0, "mean": None}

    # Step 3: count and mean of THIS column only (Series, not DataFrame).
    # FIRST ATTEMPT (wrong shape / type):
    #   df.agg(["count", "mean"]).to_dict("records")
    #     -> [{'marks': 2.0}, {'marks': 85.0}]  (whole DataFrame aggregated)
    #   s.agg(["count", "mean"]).to_dict()
    #     -> {'marks': {'count': 2.0, ...}}  (DataFrame -> nested dict)
    df1 = s[column].agg(["count", "mean"])               # CHANGED: s[column]

    # Step 4: build the dict with plain Python types.
    # count and mean share one Series, so count is a float (2.0) -> int().
    return {"count": int(df1["count"]), "mean": float(df1["mean"])}   # CHANGED


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Why count became 2.0 with agg:
#      A Series has ONE dtype. agg(["count", "mean"]) puts 2 and 85.0 in the
#      same Series -> float64 -> 2.0. Building the dict with int() / float()
#      keeps each value's own type.
#
# 2. Empty checks:
#        s.empty          -> True if no rows  (attribute, no brackets)
#        len(s) == 0      -> same idea
#        s.count() == 0   -> same here (count ignores NaN)
#
# 3. count() vs len() vs size:
#        s.count()  -> non-missing values only
#        len(s)     -> all rows, NaN included
#      After dropna they are equal; before dropna they differ.
#
# 4. Shortcut without dropna:
#        s = df[column]
#        s.count()  and  s.mean()  both skip NaN automatically,
#      but the task asks to clean first - dropna makes the step explicit.
#
# 5. Common mistakes:
#      - dropna() without subset       -> rows with other missing values lost
#      - no empty check                -> {'count': 0, 'mean': nan} (wrong)
#      - returning numpy types         -> np.int64 / np.float64, not int/float
#      - agg on the whole DataFrame    -> stats per column, wrong shape
#
# 6. Edge cases worth naming in an interview:
#      - every value None              -> {'count': 0, 'mean': None}
#      - one real value                -> count 1, mean = that value
#      - missing in ANOTHER column     -> row kept (subset)
#      - float values                  -> mean as float


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'marks': 80}, {'marks': None}, {'marks': 90}]
    big = [{'v': None if i % 4 == 0 else i} for i in range(1000)]
    vals = [i for i in range(1000) if i % 4 != 0]

    tests = [
        # (records, column, expected, description)
        (example, 'marks', {'count': 2, 'mean': 85.0}, "given example"),
        ([{'marks': None}, {'marks': None}], 'marks',
         {'count': 0, 'mean': None}, "every value missing"),
        ([{'marks': 70}], 'marks', {'count': 1, 'mean': 70.0},
         "single value"),
        ([{'n': None, 'm': 60}, {'n': 'b', 'm': 40}], 'm',
         {'count': 2, 'mean': 50.0}, "missing in ANOTHER column"),
        ([{'m': 1.5}, {'m': 2.0}], 'm', {'count': 2, 'mean': 1.75},
         "float values"),
        (big, 'v', {'count': len(vals), 'mean': sum(vals) / len(vals)},
         "max size n=1000"),
    ]

    passed = 0
    for records, column, expected, desc in tests:
        got = clean_and_summarize(records, column)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<28} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = clean_and_summarize(example, 'marks')
    assert type(out['count']) is int and type(out['mean']) is float, \
        "plain int / float"
    empty = clean_and_summarize([{'marks': None}], 'marks')
    assert empty['mean'] is None, "real None when empty"
    assert example[1]['marks'] is None and len(example) == 3, "input unchanged"
    print("PASS: plain types / None when empty / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()