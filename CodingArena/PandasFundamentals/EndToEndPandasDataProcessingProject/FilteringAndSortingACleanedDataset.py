"""
    Filtering and Sorting a Cleaned Dataset
    Easy
    End-to-End Pandas Data Processing Project
    Write full_pipeline_basic(records, column, threshold). Drop rows missing
    column, keep only rows where it's strictly greater than threshold, then
    sort what remains by column descending. Return the result as a list of
    row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 40}, {'name': 'Sam', 'marks': None},
                {'name': 'Ana', 'marks': 90}], 'marks', 50
    Returns: [{'name': 'Ana', 'marks': 90.0}]
    Explanation: Sam is dropped for missing data, Ravi's 40 doesn't clear 50,
    leaving only Ana's row.
    Constraints

    records is a non-empty list of dicts sharing the same keys, where column
    may be None. 1 <= len(records) <= 1000
"""
import pandas as pd


def full_pipeline_basic(records, column, threshold):
    """
    dropna(column) -> keep column > threshold -> sort descending,
    returned as a list of row dicts.

    Time:  O(n log n) - the sort dominates; dropna and filter are O(n).
    Space: O(n)       - for the DataFrame.
    """
    # Step 0: list of dicts -> DataFrame. None -> NaN; a number column with
    # NaN becomes float64, so 90 shows as 90.0 in the output.
    df = pd.DataFrame(records)

    # Step 1: drop rows missing THIS column only.
    # FIRST ATTEMPT (too strict): df.dropna() checks EVERY column, so a row
    # with a missing name but valid marks would be wrongly removed.
    # subset takes a LIST of columns -> [column].
    df = df.dropna(subset=[column])

    # Step 2: boolean mask. ">" is STRICT - a value equal to threshold is
    # dropped.
    df = df[df[column] > threshold]

    # Step 3: sort by column, highest first.
    # FIRST ATTEMPT (missed): no sort step at all.
    # kind="stable" keeps equal values in their original order (ties).
    df = df.sort_values(column, ascending=False, kind="stable")

    # Step 4: DataFrame -> list of row dicts ("records" is a keyword).
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Fully chained version (one expression):
#        return (pd.DataFrame(records)
#                  .dropna(subset=[column])
#                  .loc[lambda d: d[column] > threshold]
#                  .sort_values(column, ascending=False, kind="stable")
#                  .to_dict("records"))
#
# 2. dropna() vs dropna(subset=[column]):
#        dropna()                  -> drop row if ANY column is missing
#        dropna(subset=[column])   -> drop row only if THIS column is missing
#    Read the task: "Drop rows missing column" -> subset.
#
# 3. Does the order of steps matter?
#      - Filtering before dropna gives the same rows (NaN > x is False).
#      - Sorting LAST is cheaper: fewer rows left to sort.
#      - Standard workflow: clean -> filter -> sort -> return.
#
# 4. Sorting details:
#        ascending=False          -> highest first
#        kind="stable"            -> ties keep original order
#        sort_values([c1, c2], ascending=[False, True]) -> multi-column
#
# 5. Common mistakes:
#      - dropna() without subset      -> rows with other missing values lost
#      - forgetting the sort          -> example passes (1 row), hidden fail
#      - >= instead of >              -> equal values wrongly kept
#      - sort_values without "df ="   -> sort is lost (returns a new table)
#
# 6. Edge cases worth naming in an interview:
#      - nothing passes the filter     -> []
#      - missing value in ANOTHER col  -> row kept (subset); that value
#        comes back as nan, not None
#      - equal marks (ties)            -> original order kept (stable)
#      - value equal to threshold      -> dropped (strict)
#
# 7. Related: filter_and_clean (dropna -> drop_duplicates -> filter) and
#    sort_by_column - this problem chains clean, filter and sort.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'name': 'Ravi', 'marks': 40}, {'name': 'Sam', 'marks': None},
               {'name': 'Ana', 'marks': 90}]
    big = [{'id': i, 'v': None if i % 9 == 0 else i % 300}
           for i in range(1000)]
    big_expected = (pd.DataFrame(big).dropna(subset=['v'])
                      .loc[lambda d: d['v'] > 150]
                      .sort_values('v', ascending=False, kind="stable")
                      .to_dict("records"))

    tests = [
        # (records, column, threshold, expected, description)
        (example, 'marks', 50, [{'name': 'Ana', 'marks': 90.0}],
         "given example"),
        (example, 'marks', 100, [], "nothing passes"),
        ([{'n': 'a', 'm': 60}, {'n': 'b', 'm': 80}, {'n': 'c', 'm': 70}],
         'm', 50,
         [{'n': 'b', 'm': 80}, {'n': 'c', 'm': 70}, {'n': 'a', 'm': 60}],
         "sorted descending"),
        ([{'n': None, 'm': 60}, {'n': 'b', 'm': 40}], 'm', 50,
         "kept", "missing in ANOTHER column"),   # n comes back as nan
        ([{'n': 'a', 'm': 70}, {'n': 'b', 'm': 90}, {'n': 'c', 'm': 70}],
         'm', 50,
         [{'n': 'b', 'm': 90}, {'n': 'a', 'm': 70}, {'n': 'c', 'm': 70}],
         "ties keep original order"),
        ([{'m': 50}, {'m': 51}], 'm', 50, [{'m': 51}], "equal is excluded"),
        (big, 'v', 150, big_expected, "max size n=1000"),
    ]

    passed = 0
    for records, column, threshold, expected, desc in tests:
        got = full_pipeline_basic(records, column, threshold)
        if expected == "kept":   # row survives; its missing n is NaN
            ok = len(got) == 1 and got[0]['m'] == 60 and pd.isna(got[0]['n'])
        else:
            ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 1 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<28} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = full_pipeline_basic(example, 'marks', 50)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert example[1]['marks'] is None and len(example) == 3, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()