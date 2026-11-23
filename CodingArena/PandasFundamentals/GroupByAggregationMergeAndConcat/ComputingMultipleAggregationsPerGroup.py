"""
    Computing Multiple Aggregations per Group
    Medium
    GroupBy, Aggregation, Merge and Concat
    Write group_multiple_agg(records, group_col, value_col). Build a DataFrame
    and, for each distinct value of group_col, compute the min, max and mean
    of value_col in one call - .agg(['min', 'max', 'mean']) - rounding the
    mean to 4 decimals. Return {group_value: {"min": ..., "max": ...,
    "mean": ...}}.

    Examples

    Arguments: [{'category': 'A', 'amount': 10},
                {'category': 'A', 'amount': 30},
                {'category': 'B', 'amount': 5}], 'category', 'amount'
    Returns: {'A': {'min': 10, 'max': 30, 'mean': 20.0},
              'B': {'min': 5, 'max': 5, 'mean': 5.0}}
    Explanation: Group A's amounts (10, 30) give min 10, max 30, mean 20.0;
    group B has just its own single value everywhere.
    Constraints

    records is a non-empty list of dicts sharing the same keys. value_col
    holds numeric values. 1 <= len(records) <= 1000
"""
import pandas as pd


def group_multiple_agg(records, group_col, value_col):
    """
    groupby -> agg(['min', 'max', 'mean']) -> round mean -> to_dict("index").

    Time:  O(n) - one pass to group and compute the 3 stats.
    Space: O(n) - for the DataFrame (+ O(g) for g groups in the result).
    """
    # Step 0: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 1: three stats per group in ONE call. A list in SQUARE brackets
    # gives a DataFrame: index = groups, one column per stat.
    #              min  max  mean
    #     category
    #     A         10   30  20.0
    #     B          5    5   5.0
    res = df.groupby(group_col)[value_col].agg(["min", "max", "mean"])

    # Step 2: round ONLY the mean column (min/max stay ints).
    # FIRST ATTEMPT (missed): no rounding - the example passed (20.0, 5.0)
    # but a mean like 16.666666... must come back as 16.6667.
    res["mean"] = res["mean"].round(4)

    # Step 3: {group: {stat: value}} -> orient "index" (index = outer keys).
    # FIRST ATTEMPT (wrong shape): to_dict("records") -> a list of dicts and
    # the group names were LOST, because "records" drops the index.
    return res.to_dict("index")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. to_dict orients (pick by the expected output's OUTER keys):
#        "records"        -> [{col: val}, ...]       index dropped
#        "index"          -> {index: {col: val}}     groups outside  <- here
#        "dict" (default) -> {col: {index: val}}     stats outside (flipped)
#        "list"           -> {col: [values]}
#    "records" / "index" are KEYWORDS (orient), not variables.
#
# 2. Two ways to write the agg (same columns here):
#        .agg(["min", "max", "mean"])                 list -> SQUARE brackets
#        .agg(min="min", max="max", mean="mean")      named aggregation
#    Named form lets you rename: .agg(lowest="min") -> column 'lowest'.
#    agg({...}) with curly braces is for a DataFrame, mapping column -> func.
#
# 3. Rounding:
#      res["mean"].round(4) rounds one column; res.round(4) would round all
#      (harmless for int min/max here, but be specific).
#      Python's round() can't take a Series/array - use .round() / np.round.
#
# 4. Why min/max stay int but mean is float:
#      min/max pick an existing value (int); mean divides -> float.
#
# 5. Common mistakes:
#      - to_dict("records")          -> list, group names lost
#      - to_dict()                   -> {'min': {'A': 10, ...}} flipped
#      - agg("min", "max", "mean")   -> wrong; needs a LIST
#      - forgetting the rounding     -> hidden tests with long decimals fail
#
# 6. Edge cases worth naming in an interview:
#      - group with one value        -> min = max = mean
#      - repeating decimals          -> mean rounded to 4 places
#      - negative / float values     -> handled the same
#      - one group only              -> one outer key


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'category': 'A', 'amount': 10}, {'category': 'A', 'amount': 30},
               {'category': 'B', 'amount': 5}]
    big = [{'g': f"G{i % 4}", 'v': i} for i in range(1000)]
    big_expected = {}
    for k in range(4):
        vals = list(range(k, 1000, 4))
        big_expected[f"G{k}"] = {'min': min(vals), 'max': max(vals),
                                 'mean': round(sum(vals) / len(vals), 4)}

    tests = [
        # (records, group_col, value_col, expected, description)
        (example, 'category', 'amount',
         {'A': {'min': 10, 'max': 30, 'mean': 20.0},
          'B': {'min': 5, 'max': 5, 'mean': 5.0}}, "given example"),
        ([{'g': 'x', 'v': 10}, {'g': 'x', 'v': 20}, {'g': 'x', 'v': 20}],
         'g', 'v', {'x': {'min': 10, 'max': 20, 'mean': 16.6667}},
         "mean rounded to 4"),
        ([{'g': 'a', 'v': 7}], 'g', 'v',
         {'a': {'min': 7, 'max': 7, 'mean': 7.0}}, "single value"),
        ([{'g': 'a', 'v': -3}, {'g': 'a', 'v': 4}], 'g', 'v',
         {'a': {'min': -3, 'max': 4, 'mean': 0.5}}, "negative values"),
        ([{'g': 'a', 'v': 1.1}, {'g': 'a', 'v': 2.2}], 'g', 'v',
         {'a': {'min': 1.1, 'max': 2.2, 'mean': 1.65}}, "float values"),
        (big, 'g', 'v', big_expected, "max size n=1000"),
    ]

    passed = 0
    for records, group_col, value_col, expected, desc in tests:
        got = group_multiple_agg(records, group_col, value_col)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 1 else f"{{{len(got)} groups}}"
        print(f"{status}: {desc:<24} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = group_multiple_agg(example, 'category', 'amount')
    assert type(out) is dict and type(out['A']) is dict, "dict of dicts"
    assert type(out['A']['min']) is int, "min stays int"
    assert len(example) == 3, "input unchanged"
    print("PASS: return type / int min / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()