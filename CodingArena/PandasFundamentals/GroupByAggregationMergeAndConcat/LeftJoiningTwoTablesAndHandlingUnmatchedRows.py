"""
    Left-Joining Two Tables and Handling Unmatched Rows
    Hard
    GroupBy, Aggregation, Merge and Concat
    Write left_merge_with_missing(records_a, records_b, on). Join records_a
    with records_b on on using how="left", so EVERY row of records_a is kept
    even without a match. Convert any resulting missing values to None before
    returning the joined table as a list of row dicts.

    Examples

    Arguments: [{'id': 1, 'name': 'Ravi'}, {'id': 2, 'name': 'Sam'}],
               [{'id': 1, 'amount': 500}], 'id'
    Returns: [{'id': 1, 'name': 'Ravi', 'amount': 500.0},
              {'id': 2, 'name': 'Sam', 'amount': None}]
    Explanation: Ravi (id 1) picks up an amount of 500; Sam (id 2) has no
    match, so his amount comes back as None instead of being dropped.
    Constraints

    records_a and records_b are non-empty lists of dicts. Both contain the
    column named by on. 1 <= len(records_a), len(records_b) <= 500
"""
import pandas as pd


def left_merge_with_missing(records_a, records_b, on):
    """
    left merge on `on` -> NaN to None -> list of row dicts.

    Time:  O(n + m) - hash join over n rows of A and m rows of B
           (more if a key repeats in B and rows multiply).
    Space: O(n + m) - for the DataFrames and the result.
    """
    # Step 0: both lists of dicts -> DataFrames.
    df1 = pd.DataFrame(records_a)
    df2 = pd.DataFrame(records_b)

    # Step 1: LEFT join - every row of df1 stays, in df1's order.
    # Rows with no match in df2 get NaN in df2's columns, and a number
    # column with NaN becomes float64, so 500 shows as 500.0.
    df3 = df1.merge(df2, on=on, how="left")

    # Step 2: NaN -> None.
    # FIRST ATTEMPT (wrong output): returned df3.to_dict("records") directly
    # -> Sam's amount came back as nan, not None.
    # Also: df3.where(df3.notna(), None) alone is NOT enough - a float column
    # can't hold None, so pandas turns it straight back into NaN.
    # astype(object) first lets each column hold any Python value (incl. None);
    # where(cond, None) keeps values where cond is True, puts None elsewhere.
    df3 = df3.astype(object).where(df3.notna(), None)

    # Step 3: DataFrame -> list of row dicts ("records" is a keyword).
    return df3.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. The four join types (how=...):
#        "inner" -> only keys in BOTH tables         (Sam would be dropped)
#        "left"  -> all rows of LEFT, matches from right  (this problem)
#        "right" -> all rows of RIGHT, matches from left
#        "outer" -> all rows of both, NaN where no match
#
# 2. Why NaN and not None after a merge?
#      pandas marks missing numbers as NaN (a float). That is why the int
#      column 'amount' becomes float: 500 -> 500.0. To return None we must
#      change the column type to object first.
#
# 3. Other ways to do NaN -> None (same idea, object type needed):
#        df3.astype(object).where(df3.notna(), None)      <- used here
#        df3.replace({np.nan: None})                      (needs numpy;
#            result depends on pandas version - check the output)
#      fillna(None) does NOT work - fillna needs a real value.
#
# 4. Duplicate keys in the right table:
#      If records_b has id 1 twice, Ravi's row appears TWICE in the result
#      (one per match). A left join keeps every left row but can multiply
#      them. Check with len(df3) vs len(df1).
#
# 5. Same column name in both tables (not the key):
#      pandas adds suffixes: name_x (left), name_y (right).
#      Change them with suffixes=("_a", "_b").
#
# 6. Common mistakes:
#      - how="inner" (the default!)         -> unmatched rows disappear
#      - returning without NaN -> None       -> nan in the output
#      - where(..., None) without astype     -> NaN stays NaN
#      - merge(df2, "id") positional         -> works, but on=on is clearer
#
# 7. Edge cases worth naming in an interview:
#      - every row matches               -> no None at all
#      - no row matches                  -> all right-side columns None
#      - missing value already in A      -> also becomes None
#      - key repeated in B               -> left row duplicated
#      - text column from B unmatched    -> None too (not just numbers)
#
# 8. Related: Midstone 2 Q3 (unknown restaurant) - restaurant_id is the
#    shared key; ~isin finds the rows a left join would leave with NaN.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    a = [{'id': 1, 'name': 'Ravi'}, {'id': 2, 'name': 'Sam'}]
    b = [{'id': 1, 'amount': 500}]
    big_a = [{'k': i, 'x': i * 2} for i in range(500)]
    big_b = [{'k': i, 'y': i} for i in range(0, 500, 2)]

    tests = [
        # (records_a, records_b, on, expected, description)
        (a, b, 'id',
         [{'id': 1, 'name': 'Ravi', 'amount': 500.0},
          {'id': 2, 'name': 'Sam', 'amount': None}], "given example"),
        (a, [{'id': 1, 'amount': 5}, {'id': 2, 'amount': 7}], 'id',
         [{'id': 1, 'name': 'Ravi', 'amount': 5},
          {'id': 2, 'name': 'Sam', 'amount': 7}], "every row matches"),
        (a, [{'id': 9, 'amount': 1}], 'id',
         [{'id': 1, 'name': 'Ravi', 'amount': None},
          {'id': 2, 'name': 'Sam', 'amount': None}], "no row matches"),
        ([{'id': 1, 'n': None}], [{'id': 1, 'v': 3}], 'id',
         [{'id': 1, 'n': None, 'v': 3}], "missing value already in A"),
        ([{'id': 1}], [{'id': 1, 'v': 3}, {'id': 1, 'v': 4}], 'id',
         [{'id': 1, 'v': 3}, {'id': 1, 'v': 4}], "key repeated in B"),
        (a, [{'id': 1, 'city': 'Chennai'}], 'id',
         [{'id': 1, 'name': 'Ravi', 'city': 'Chennai'},
          {'id': 2, 'name': 'Sam', 'city': None}], "unmatched text column"),
        (big_a, big_b, 'k', None, "max size n=500"),
    ]

    passed = 0
    for records_a, records_b, on, expected, desc in tests:
        got = left_merge_with_missing(records_a, records_b, on)
        if expected is None:   # big case: check shape and None count
            ok = (len(got) == 500
                  and sum(r['y'] is None for r in got) == 250)
        else:
            ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<30} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = left_merge_with_missing(a, b, 'id')
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert out[1]['amount'] is None, "real None, not nan"
    assert len(a) == 2 and 'amount' not in a[0], "input unchanged"
    print("PASS: return type / real None / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()