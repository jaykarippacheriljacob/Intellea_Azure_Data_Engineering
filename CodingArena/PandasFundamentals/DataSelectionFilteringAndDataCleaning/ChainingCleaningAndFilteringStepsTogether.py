"""
    Chaining Cleaning and Filtering Steps Together
    Hard
    Data Selection, Filtering and Data Cleaning
    Write filter_and_clean(records, column, threshold). Build a DataFrame and
    run THREE steps in order: drop rows with any missing value, drop duplicate
    rows, then keep only rows where column is strictly greater than threshold.
    Return the result as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Ravi', 'marks': 88},
                {'name': 'Sam', 'marks': None}, {'name': 'Ana', 'marks': 40}],
               'marks', 50
    Returns: [{'name': 'Ravi', 'marks': 88.0}]
    Explanation: The duplicate Ravi row collapses to one, Sam is dropped for a
    missing value, and Ana's 40 doesn't clear 50 - leaving only Ravi.
    Constraints

    records is a non-empty list of dicts sharing the same keys, where any
    value may be None. column holds numeric values (or None).
    1 <= len(records) <= 500
"""
import pandas as pd


def filter_and_clean(records, column, threshold):
    """
    dropna -> drop_duplicates -> keep rows with column > threshold,
    returned as a list of row dicts.

    Time:  O(n * k) - each step is one pass over the table.
    Space: O(n * k) - for the DataFrame.
    """
    # Step 0: list of dicts -> DataFrame. None -> NaN; a number column with
    # NaN becomes float64, so 88 shows as 88.0 in the output.
    df = pd.DataFrame(records)

    # Steps 1 + 2 chained: each method returns a NEW DataFrame, so the next
    # method runs on the result of the previous one (left to right).
    #   dropna()          -> remove rows with ANY missing value
    #   drop_duplicates() -> remove repeated rows (keep the first)
    # FIRST ATTEMPT (error): df.duplicates() -> AttributeError. The method
    # is drop_duplicates() (removes); duplicated() only marks True/False.
    df = df.dropna().drop_duplicates()

    # Step 3: boolean mask. ">" is STRICT - a value equal to threshold is
    # dropped.
    df = df[df[column] > threshold]

    # Step 4: DataFrame -> list of row dicts ("records" is a keyword).
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Fully chained version (one expression):
#        return (pd.DataFrame(records)
#                  .dropna()
#                  .drop_duplicates()
#                  .loc[lambda d: d[column] > threshold]
#                  .to_dict("records"))
#    .loc[lambda d: ...] lets the filter use the CURRENT table inside a
#    chain; the brackets around the whole chain allow line breaks.
#
# 2. Does the ORDER of the steps matter?
#      - dropna and drop_duplicates: same final rows either way here.
#      - Filtering also happens to give the same rows (NaN > x is False,
#        so missing rows fail the filter anyway).
#      - But the task says "in order", and on real data order matters:
#        e.g. cleaning text BEFORE drop_duplicates changes what counts as a
#        duplicate, and fillna BEFORE a filter lets filled rows pass.
#      - Cleaning first, filtering last is the standard workflow
#        (Pandas notes, section 2: inspect -> clean -> filter -> re-check).
#
# 3. Common mistakes:
#      - df.duplicates() -> AttributeError (no such method)
#      - df.drop.duplicates() -> AttributeError (drop is a method, not a
#        namespace)
#      - forgetting "df =" on any step -> that step is lost
#      - using >= instead of >  -> equal values wrongly kept
#
# 4. Edge cases worth naming in an interview:
#      - nothing passes the filter       -> []
#      - every row has a None            -> []
#      - a None in ANOTHER column        -> row dropped too (dropna checks
#        every column)
#      - duplicates that only differ by a missing value -> the one with
#        the None is removed by dropna, the other stays
#      - value equal to threshold        -> dropped (strict)
#
# 5. Related: remove_duplicate_rows, drop_missing_rows and
#    filter_by_condition - this problem chains all three.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'name': 'Ravi', 'marks': 88}, {'name': 'Ravi', 'marks': 88},
               {'name': 'Sam', 'marks': None}, {'name': 'Ana', 'marks': 40}]
    big = [{'id': i % 200, 'v': None if i % 7 == 0 else (i % 200)}
           for i in range(500)]
    big_expected = (pd.DataFrame(big).dropna().drop_duplicates()
                      .loc[lambda d: d['v'] > 100].to_dict("records"))

    tests = [
        # (records, column, threshold, expected, description)
        (example, 'marks', 50, [{'name': 'Ravi', 'marks': 88.0}],
         "given example"),
        (example, 'marks', 100, [], "nothing passes"),
        ([{'a': None}, {'a': None}], 'a', 0, [], "every row missing"),
        ([{'m': 60, 'c': None}, {'m': 70, 'c': 'x'}], 'm', 50,
         [{'m': 70, 'c': 'x'}], "None in another column"),
        ([{'m': 50}, {'m': 51}], 'm', 50, [{'m': 51}], "equal is excluded"),
        ([{'m': 5}, {'m': 9}, {'m': 5}, {'m': 9}], 'm', 0,
         [{'m': 5}, {'m': 9}], "duplicates kept once, order kept"),
        ([{'m': -1}, {'m': -5}], 'm', -3, [{'m': -1}], "negative threshold"),
        (big, 'v', 100, big_expected, "max size n=500"),
    ]

    passed = 0
    for records, column, threshold, expected, desc in tests:
        got = filter_and_clean(records, column, threshold)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<32} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = filter_and_clean(example, 'marks', 50)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert example[2]['marks'] is None and len(example) == 4, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()
    