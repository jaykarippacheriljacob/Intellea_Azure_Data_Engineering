"""
    Merging Two Tables Then Aggregating the Combined Result
    Medium
    End-to-End Pandas Data Processing Project
    Write merge_clean_and_aggregate(records_a, records_b, on). Inner-join the
    two tables on on, drop any rows with missing values, then return
    {"row_count": ..., "total_amount": ...} - the number of surviving rows and
    the sum of their amount column. If nothing survives, return
    {"row_count": 0, "total_amount": 0.0}.

    Examples

    Arguments: [{'id': 1, 'name': 'Ravi'}, {'id': 2, 'name': 'Sam'}],
               [{'id': 1, 'amount': 100}, {'id': 2, 'amount': 200}], 'id'
    Returns: {'row_count': 2, 'total_amount': 300.0}
    Explanation: Both ids match and merge cleanly: row_count 2,
    total_amount 300.0.
    Constraints

    records_a and records_b are non-empty lists of dicts. records_b always
    has an amount column. Both contain the column named by on.
    1 <= len(records_a), len(records_b) <= 500
"""
import pandas as pd


def merge_clean_and_aggregate(records_a, records_b, on):
    """
    inner merge -> dropna -> count + sum of amount -> plain-type dict.

    Time:  O(n + m) - hash join, then one pass to clean and aggregate.
    Space: O(n + m) - for the DataFrames.
    """
    # Step 0: both lists of dicts -> DataFrames.
    df1 = pd.DataFrame(records_a)
    df2 = pd.DataFrame(records_b)

    # Step 1: INNER join - keep only ids found in BOTH tables.
    df3 = df1.merge(df2, on=on, how="inner")

    # Step 2: drop rows with a missing value in ANY column. Plain dropna()
    # is right here: the task says "drop any rows with missing values"
    # (compare clean_and_summarize, which needed subset=[column]).
    df3 = df3.dropna()

    # Step 3: two stats on the amount column in one call (named aggregation:
    # new_name="function"). Result is a Series:
    #     row_count       2
    #     total_amount  300
    df4 = df3["amount"].agg(
        row_count="count",
        total_amount="sum")

    # Step 4: build the dict with plain Python types.
    # FIRST ATTEMPT (type): df4.to_dict() -> {'row_count': 2,
    # 'total_amount': 300} - int 300, but the task shows float 300.0.
    # 300 == 300.0 in Python, but float() makes the type match exactly.
    # Empty case needs no special check: count -> 0, sum -> 0 -> 0.0.
    return {"row_count": int(df4["row_count"]),
            "total_amount": float(df4["total_amount"])}


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Simpler version without agg:
#        return {"row_count": len(df3),
#                "total_amount": float(df3["amount"].sum())}
#    len(df3) = rows left after dropna (same as count here).
#
# 2. Why no empty check is needed (unlike clean_and_summarize):
#        sum() of nothing   -> 0     (a real number)
#        count() of nothing -> 0
#        mean() of nothing  -> NaN   (that's why the other problem needed
#                                     a check for None)
#
# 3. dropna() vs dropna(subset=[...]) - read the task:
#        "drop any rows with missing values"   -> dropna()
#        "drop rows where column is missing"   -> dropna(subset=[column])
#
# 4. Order matters here: merge FIRST, then dropna.
#      A left join would create NaN for unmatched ids, which dropna would
#      then remove - the same rows an inner join drops. Inner join is the
#      direct way.
#
# 5. Common mistakes:
#      - how="left"                     -> extra NaN rows (dropna fixes it,
#                                          but inner is the task)
#      - returning df4.to_dict()        -> numpy-backed ints, 300 not 300.0
#      - summing before dropna          -> includes rows that should go
#      - dropna(subset=["amount"])      -> keeps rows missing a name
#
# 6. Edge cases worth naming in an interview:
#      - no matching ids                -> {'row_count': 0, 'total_amount': 0.0}
#      - missing name in a matched row  -> that row dropped
#      - missing amount                 -> that row dropped
#      - key repeated in B              -> left row counted twice


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    a = [{'id': 1, 'name': 'Ravi'}, {'id': 2, 'name': 'Sam'}]
    b = [{'id': 1, 'amount': 100}, {'id': 2, 'amount': 200}]
    big_a = [{'id': i, 'name': None if i % 10 == 0 else f"n{i}"}
             for i in range(500)]
    big_b = [{'id': i, 'amount': i} for i in range(0, 500, 2)]
    kept = [i for i in range(0, 500, 2) if i % 10 != 0]

    tests = [
        # (records_a, records_b, on, expected, description)
        (a, b, 'id', {'row_count': 2, 'total_amount': 300.0},
         "given example"),
        (a, [{'id': 9, 'amount': 50}], 'id',
         {'row_count': 0, 'total_amount': 0.0}, "no matching ids"),
        ([{'id': 1, 'name': None}, {'id': 2, 'name': 'Sam'}], b, 'id',
         {'row_count': 1, 'total_amount': 200.0}, "missing name dropped"),
        (a, [{'id': 1, 'amount': None}, {'id': 2, 'amount': 200}], 'id',
         {'row_count': 1, 'total_amount': 200.0}, "missing amount dropped"),
        ([{'id': 1, 'name': 'Ravi'}],
         [{'id': 1, 'amount': 10}, {'id': 1, 'amount': 15}], 'id',
         {'row_count': 2, 'total_amount': 25.0}, "key repeated in B"),
        (big_a, big_b, 'id',
         {'row_count': len(kept), 'total_amount': float(sum(kept))},
         "max size n=500"),
    ]

    passed = 0
    for records_a, records_b, on, expected, desc in tests:
        got = merge_clean_and_aggregate(records_a, records_b, on)
        ok = (got == expected
              and type(got['total_amount']) is float
              and type(got['row_count']) is int)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<24} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    merge_clean_and_aggregate(a, b, 'id')
    assert len(a) == 2 and 'amount' not in a[0], "input unchanged"
    print("PASS: input untouched")

    print(f"\n{passed}/{len(tests)} value + type tests passed")


if __name__ == "__main__":
    run_tests()