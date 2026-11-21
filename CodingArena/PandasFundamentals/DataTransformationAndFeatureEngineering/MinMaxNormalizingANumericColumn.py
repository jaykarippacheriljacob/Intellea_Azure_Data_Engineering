"""
    Min-Max Normalizing a Numeric Column
    Medium
    Data Transformation and Feature Engineering
    Write normalize_column(records, column). Build a DataFrame and add a
    normalized column scaling column into the 0-1 range with
    (value - min) / (max - min), rounded to 4 decimals. If every value is
    identical (max equals min), set every normalized value to 0.0 instead of
    dividing by zero. Return the full table as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 0}, {'name': 'Sam', 'marks': 50},
                {'name': 'Ana', 'marks': 100}], 'marks'
    Returns: [{'name': 'Ravi', 'marks': 0, 'normalized': 0.0},
              {'name': 'Sam', 'marks': 50, 'normalized': 0.5},
              {'name': 'Ana', 'marks': 100, 'normalized': 1.0}]
    Explanation: 0 becomes 0.0, 100 becomes 1.0, and 50 lands exactly halfway
    at 0.5.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds
    numeric values. 1 <= len(records) <= 500
"""
import pandas as pd


def normalize_column(records, column):
    """
    Add `normalized` = (value - min) / (max - min), rounded to 4 decimals,
    or 0.0 everywhere if all values are equal. Return all rows as dicts.

    Time:  O(n * k) - building the DataFrame; min, max, maths are O(n).
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 2: min and max of the column (computed once, reused below).
    dfmin = df[column].min()
    dfmax = df[column].max()

    if dfmax == dfmin:
        # All values identical -> max - min = 0 -> dividing would give NaN.
        # REVISION 1 (bug fix): the first version used
        #     round(df[column] - dfmax, 4)
        # which keeps the column's type: an int column gives int 0, not the
        # required 0.0. Assigning a single value fills the whole column.
        df["normalized"] = 0.0
    else:
        # Vectorised: the whole column at once, no loop.
        # round(Series, 4) works (same as .round(4)).
        df["normalized"] = ((df[column] - dfmin) / (dfmax - dfmin)).round(4)

    # Step 3: FULL table (original columns + normalized) -> row dicts.
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas):
#        vals = [r[column] for r in records]
#        lo, hi = min(vals), max(vals)
#        return [{**r, "normalized": 0.0 if hi == lo
#                 else round((r[column] - lo) / (hi - lo), 4)}
#                for r in records]
#
# 2. Why check max == min first:
#        (x - min) / 0 -> pandas gives NaN (0/0) or inf, NOT an error,
#        so the bug would be silent. Always guard the division.
#
# 3. Other scaling methods (feature engineering):
#        min-max:  (x - min) / (max - min)        -> 0 to 1
#        z-score:  (x - mean) / std               -> mean 0, std 1
#        percent of max: x / max
#    Min-max is sensitive to outliers: one huge value squeezes the rest
#    close to 0.
#
# 4. Rounding note: pandas .round() / round(Series) round exact halves to
#    the nearest EVEN digit (0.00005 -> 0.0 at 4 decimals), same as
#    Python's round().
#
# 5. Edge cases worth naming in an interview:
#      - single row              -> max == min -> 0.0
#      - all values equal        -> 0.0 for every row (float, not int)
#      - negative values         -> still 0 to 1
#      - unsorted input          -> order kept, min -> 0.0, max -> 1.0
#      - original column         -> unchanged (0 stays 0, not 0.0)
#
# 6. Related: NumPy Fundamentals E6 (min-max normalise each column with
#    broadcasting), add_computed_column (new column from arithmetic).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'id': i, 'v': i} for i in range(500)]

    tests = [
        # (records, column, expected, description)
        ([{'name': 'Ravi', 'marks': 0}, {'name': 'Sam', 'marks': 50},
          {'name': 'Ana', 'marks': 100}], 'marks',
         [{'name': 'Ravi', 'marks': 0, 'normalized': 0.0},
          {'name': 'Sam', 'marks': 50, 'normalized': 0.5},
          {'name': 'Ana', 'marks': 100, 'normalized': 1.0}],
         "given example"),
        ([{'m': 5}, {'m': 5}], 'm',
         [{'m': 5, 'normalized': 0.0}, {'m': 5, 'normalized': 0.0}],
         "all identical -> 0.0"),
        ([{'m': 7}], 'm', [{'m': 7, 'normalized': 0.0}], "single row"),
        ([{'m': 0}, {'m': 1}, {'m': 3}], 'm',
         [{'m': 0, 'normalized': 0.0}, {'m': 1, 'normalized': 0.3333},
          {'m': 3, 'normalized': 1.0}], "rounded to 4 decimals"),
        ([{'m': -10}, {'m': 0}, {'m': 10}], 'm',
         [{'m': -10, 'normalized': 0.0}, {'m': 0, 'normalized': 0.5},
          {'m': 10, 'normalized': 1.0}], "negative values"),
        ([{'m': 30}, {'m': 10}, {'m': 20}], 'm',
         [{'m': 30, 'normalized': 1.0}, {'m': 10, 'normalized': 0.0},
          {'m': 20, 'normalized': 0.5}], "unsorted input"),
        ([{'p': 1.5}, {'p': 2.5}], 'p',
         [{'p': 1.5, 'normalized': 0.0}, {'p': 2.5, 'normalized': 1.0}],
         "float column"),
        (big, 'v', [{'id': i, 'v': i, 'normalized': round(i / 499, 4)}
                    for i in range(500)], "max size n=500"),
    ]

    passed = 0
    for records, column, expected, desc in tests:
        got = normalize_column(records, column)
        ok = got == expected and all(type(r['normalized']) is float for r in got)
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 3 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<24} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [{'name': 'Ravi', 'marks': 0}, {'name': 'Sam', 'marks': 50}]
    out = normalize_column(data, 'marks')
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert list(out[0]) == ['name', 'marks', 'normalized'], "new column last"
    assert 'normalized' not in data[0], "input unchanged"
    print("PASS: return type / column order / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed (incl. float type check)")


if __name__ == "__main__":
    run_tests()
    