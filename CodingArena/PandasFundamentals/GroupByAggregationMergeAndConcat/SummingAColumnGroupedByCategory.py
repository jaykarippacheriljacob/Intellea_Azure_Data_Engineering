"""
    Summing a Column Grouped by Category
    Easy
    GroupBy, Aggregation, Merge and Concat
    Write group_sum(records, group_col, value_col). Build a DataFrame and
    return the total of value_col for each distinct value of group_col, using
    df.groupby(group_col)[value_col].sum(), as a {group_value: total} dict.

    Examples

    Arguments: [{'category': 'A', 'amount': 10}, {'category': 'B', 'amount': 5},
                {'category': 'A', 'amount': 20}], 'category', 'amount'
    Returns: {'A': 30, 'B': 5}
    Explanation: A's two amounts (10 + 20) sum to 30; B has just one, 5.
    Constraints

    records is a non-empty list of dicts sharing the same keys. value_col
    holds numeric values.
    1 <= len(records) <= 1000
"""
import pandas as pd


def group_sum(records, group_col, value_col):
    """
    groupby(group_col) -> pick value_col -> sum each group,
    returned as a {group_value: total} dict.

    Time:  O(n) - one pass to group and add up the n rows.
    Space: O(n) - for the DataFrame (+ O(g) for g groups in the result).
    """
    # Step 0: list of dicts -> DataFrame (one row per dict).
    df = pd.DataFrame(records)

    # Step 1: split the rows by group_col, take ONLY value_col, add each group.
    # The result is a SERIES, not a DataFrame:
    #     category
    #     A    30        <- index = group values
    #     B     5        <- values = totals
    # Groups come out sorted by key (groupby default sort=True).
    total = df.groupby(group_col)[value_col].sum()

    # Step 2: Series -> {index: value} dict.
    # FIRST ATTEMPT (error): total.to_dict("records") -> TypeError:
    # unsupported type: <class 'str'>. "records" is a keyword for a
    # DATAFRAME only; a Series' to_dict() takes no orient, so it is just
    # to_dict().
    return total.to_dict()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Series.to_dict() vs DataFrame.to_dict("records"):
#        Series.to_dict()            -> {index: value}           {'A': 30, 'B': 5}
#        Series.to_dict("records")   -> ERROR
#        DataFrame.to_dict("records")-> [{col: value, ...}, ...] one dict per row
#    Rule: look at the expected output.
#        {...}            -> Series.to_dict()
#        [{...}, {...}]   -> reset_index() first, then to_dict("records")
#
# 2. If a list of row dicts were wanted instead:
#        total.reset_index().to_dict("records")
#        -> [{'category': 'A', 'amount': 30}, {'category': 'B', 'amount': 5}]
#    reset_index() moves the group labels from the index back into a column,
#    turning the Series into a DataFrame.
#
# 3. Why [value_col] after groupby?
#      - df.groupby(group_col).sum()             -> sums EVERY other column
#        (a DataFrame), which can fail or add junk if there are text columns.
#      - df.groupby(group_col)[value_col].sum()  -> only the column we need
#        (a Series). Brackets = pick the column before aggregating.
#
# 4. Other aggregations, same pattern:
#        .mean()  .count()  .max()  .min()  .size()
#        .agg(["sum", "mean"])        -> list in SQUARE brackets
#        .agg(total=("amount", "sum")) -> named aggregation
#
# 5. Common mistakes:
#      - to_dict("records") on a Series      -> TypeError (unsupported type)
#      - df.groupby[group_col]               -> TypeError (groupby is a method,
#        needs round brackets)
#      - forgetting [value_col]              -> sums all columns, wrong shape
#      - returning the Series itself         -> not a plain dict
#
# 6. Edge cases worth naming in an interview:
#      - only one record                 -> {group: that value}
#      - only one group                  -> one key
#      - float values                    -> float totals (1.5 + 2.5 = 4.0)
#      - negative values                 -> summed normally
#      - numeric group keys              -> int keys in the dict {1: 8, 2: 4}
#
# 7. Related: aggregate_by_group (named agg) and the Midstone 2
#    groupby("order_id").count() > 1 check - same groupby -> column -> agg
#    pattern.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'category': 'A', 'amount': 10}, {'category': 'B', 'amount': 5},
               {'category': 'A', 'amount': 20}]
    big = [{'g': f"G{i % 10}", 'v': i} for i in range(1000)]
    big_expected = {f"G{k}": sum(range(k, 1000, 10)) for k in range(10)}

    tests = [
        # (records, group_col, value_col, expected, description)
        (example, 'category', 'amount', {'A': 30, 'B': 5}, "given example"),
        ([{'city': 'X', 'sales': 7}], 'city', 'sales', {'X': 7},
         "single record"),
        ([{'g': 'a', 'v': 1}, {'g': 'a', 'v': 2}, {'g': 'a', 'v': 3}],
         'g', 'v', {'a': 6}, "single group"),
        ([{'g': 'a', 'v': 1.5}, {'g': 'a', 'v': 2.5}, {'g': 'b', 'v': 2.0}],
         'g', 'v', {'a': 4.0, 'b': 2.0}, "float values"),
        ([{'g': 'a', 'v': -4}, {'g': 'a', 'v': 1}, {'g': 'b', 'v': -2}],
         'g', 'v', {'a': -3, 'b': -2}, "negative values"),
        ([{'k': 1, 'v': 3}, {'k': 2, 'v': 4}, {'k': 1, 'v': 5}],
         'k', 'v', {1: 8, 2: 4}, "numeric group keys"),
        ([{'g': 'x', 'v': 1, 'note': 'hi'}, {'g': 'x', 'v': 2, 'note': 'yo'}],
         'g', 'v', {'x': 3}, "extra text column ignored"),
        (big, 'g', 'v', big_expected, "max size n=1000"),
    ]

    passed = 0
    for records, group_col, value_col, expected, desc in tests:
        got = group_sum(records, group_col, value_col)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"{{{len(got)} groups}}"
        print(f"{status}: {desc:<32} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = group_sum(example, 'category', 'amount')
    assert type(out) is dict, "returns a plain dict, not a Series"
    assert len(example) == 3 and example[0]['amount'] == 10, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()