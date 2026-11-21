"""
    Dropping Rows with Missing Values
    Easy
    Data Selection, Filtering and Data Cleaning
    Write drop_missing_rows(records). Build a DataFrame (a missing value in
    records is written as None) and return only the rows with NO missing
    values at all, using df.dropna().

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': None}]
    Returns: [{'name': 'Ravi', 'marks': 88.0}]
    Explanation: Sam's row has a missing marks value and is dropped, leaving
    only Ravi's row.
    Constraints

    records is a non-empty list of dicts sharing the same keys, where any
    value may be None. 1 <= len(records) <= 500
"""
import pandas as pd


def drop_missing_rows(records):
    """
    Return only the rows of `records` with no missing value in any column,
    as a list of row dicts.

    Time:  O(n * k) - checks every cell once.
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame. pandas turns None into a missing
    # value: NaN in a number column, None/NaN in a text column.
    # A number column WITH a missing value becomes float64 -> 88 becomes 88.0
    # (that is why the expected output shows 'marks': 88.0).
    df = pd.DataFrame(records)

    # Step 2: dropna() removes every row with AT LEAST ONE missing value
    # (default how="any"). It returns a NEW DataFrame - assign it back.
    df = df.dropna()

    # Step 3: DataFrame -> list of row dicts ("records" is a keyword).
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas) - note it keeps 88 as an int, so it
#    would NOT match the expected 88.0:
#        return [r for r in records if all(v is not None for v in r.values())]
#
# 2. dropna options:
#        df.dropna()                    drop rows with ANY missing value
#        df.dropna(how="all")           drop rows where ALL values are missing
#        df.dropna(subset=["marks"])    only look at some columns
#        df.dropna(thresh=2)            keep rows with at least 2 values
#        df.dropna(axis=1)              drop COLUMNS with missing values
#
# 3. Check how much you lose first (from the notes):
#        df.isna().sum()                      missing per column
#        len(df), len(df.dropna())            rows before / after
#
# 4. Why 88 becomes 88.0:
#        NaN is a float, so a number column with any NaN is stored as
#        float64. Even after the NaN row is dropped, the column stays float.
#        Fix if ints are needed: df["marks"] = df["marks"].astype(int)
#        (only AFTER the NaN rows are gone).
#
# 5. Edge cases worth naming in an interview:
#      - no missing values        -> same rows back (no change)
#      - every row has a None     -> [] (empty list)
#      - None in a text column    -> also treated as missing, row dropped
#      - empty string ''          -> NOT missing, row kept
#      - 0                        -> NOT missing, row kept
#
# 6. Related problems: LeetCode 2883 (Drop Missing Data),
#    LeetCode 2887 (Fill Missing Data) - fillna instead of dropna.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'id': i, 'v': None if i % 5 == 0 else i} for i in range(500)]

    tests = [
        # (records, expected, description)
        ([{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': None}],
         [{'name': 'Ravi', 'marks': 88.0}], "given example"),
        ([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}],
         [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], "no missing values"),
        ([{'a': None}, {'a': None}], [], "every row missing"),
        ([{'n': None, 'm': 5}, {'n': 'Sam', 'm': 7}],
         [{'n': 'Sam', 'm': 7}], "None in a text column"),
        ([{'n': '', 'm': 0}, {'n': 'Ana', 'm': None}],
         [{'n': '', 'm': 0.0}], "'' and 0 are not missing"),
        ([{'a': None, 'b': None}, {'a': 1, 'b': 2}],
         [{'a': 1.0, 'b': 2.0}], "all-missing row dropped"),
        (big, [{'id': i, 'v': float(i)} for i in range(500) if i % 5],
         "max size n=500"),
    ]

    passed = 0
    for records, expected, desc in tests:
        got = drop_missing_rows(records)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<26} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': None}]
    out = drop_missing_rows(data)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert type(out[0]['marks']) is float, "88 -> 88.0 (column became float)"
    assert data[1] == {'name': 'Sam', 'marks': None}, "input unchanged"
    print("PASS: return type / float column / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()