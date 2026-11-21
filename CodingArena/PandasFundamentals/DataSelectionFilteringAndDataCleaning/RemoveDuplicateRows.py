"""
    Removing Duplicate Rows
    Medium
    Data Selection, Filtering and Data Cleaning
    Write remove_duplicate_rows(records). Build a DataFrame and return it with
    every duplicate row removed (keeping the FIRST occurrence), using
    df.drop_duplicates(), as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Ravi', 'marks': 88},
                {'name': 'Sam', 'marks': 90}]
    Returns: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 90}]
    Explanation: Ravi's row is repeated twice but only kept once; Sam's row
    is unique and kept.
    Constraints

    records is a non-empty list of dicts sharing the same keys.
    1 <= len(records) <= 500
"""
import pandas as pd


def remove_duplicate_rows(records):
    """
    Return `records` without repeated rows (first occurrence kept), in the
    original order, as a list of row dicts.

    Time:  O(n * k) - hashing every row once.
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 2: drop_duplicates() removes rows that are identical in EVERY
    # column. keep="first" is the default, so the first copy stays and the
    # original order is kept.
    # It returns a NEW DataFrame - assign it back (df = ...), otherwise
    # nothing changes.
    df = df.drop_duplicates()

    # Step 3: DataFrame -> list of row dicts ("records" is a keyword).
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas), keeps the first occurrence and order:
#        seen, out = set(), []
#        for row in records:
#            key = tuple(row.items())
#            if key not in seen:
#                seen.add(key)
#                out.append(row)
#        return out
#
# 2. drop_duplicates options:
#        df.drop_duplicates()                    all columns, keep first
#        df.drop_duplicates(keep="last")         keep the last copy
#        df.drop_duplicates(keep=False)          drop EVERY copy
#        df.drop_duplicates(subset=["name"])     judge by some columns only
#        df.drop_duplicates(ignore_index=True)   renumber the index 0, 1, 2
#
# 3. Count before removing (what we did with orders_raw.csv):
#        df.duplicated().sum()     -> 3 in orders
#        df.duplicated(keep=False) -> marks ALL copies, to look at them
#
# 4. Common mistakes:
#      - df.drop_duplicates() without "df =" -> df still has duplicates
#      - expecting near-duplicates to go: 'Ravi' vs 'ravi ' are DIFFERENT;
#        clean text first (.str.strip().str.title())
#      - 88 vs 88.0 in the same column: pandas makes both float -> equal
#
# 5. Edge cases worth naming in an interview:
#      - no duplicates          -> same list back
#      - all rows identical     -> one row
#      - same name, other marks -> NOT a duplicate (every column must match)
#      - duplicates not next to each other -> still removed
#
# 6. Related problems: LeetCode 2882 (Drop Duplicate Rows),
#    LeetCode 196 (Delete Duplicate Emails).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'id': i % 250, 'v': (i % 250) * 2} for i in range(500)]

    tests = [
        # (records, expected, description)
        ([{'name': 'Ravi', 'marks': 88}, {'name': 'Ravi', 'marks': 88},
          {'name': 'Sam', 'marks': 90}],
         [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 90}],
         "given example"),
        ([{'a': 1}, {'a': 2}], [{'a': 1}, {'a': 2}], "no duplicates"),
        ([{'a': 1}, {'a': 1}, {'a': 1}], [{'a': 1}], "all identical"),
        ([{'n': 'Ravi', 'm': 88}, {'n': 'Ravi', 'm': 90}],
         [{'n': 'Ravi', 'm': 88}, {'n': 'Ravi', 'm': 90}],
         "same name, other marks"),
        ([{'a': 1}, {'a': 2}, {'a': 1}, {'a': 3}, {'a': 2}],
         [{'a': 1}, {'a': 2}, {'a': 3}], "duplicates not adjacent"),
        ([{'n': 'Ravi'}, {'n': 'ravi '}], [{'n': 'Ravi'}, {'n': 'ravi '}],
         "near-duplicate text kept"),
        ([{'a': 1}], [{'a': 1}], "single row"),
        (big, [{'id': i, 'v': i * 2} for i in range(250)], "max size n=500"),
    ]

    passed = 0
    for records, expected, desc in tests:
        got = remove_duplicate_rows(records)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 3 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<26} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [{'a': 1}, {'a': 1}]
    out = remove_duplicate_rows(data)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert data == [{'a': 1}, {'a': 1}], "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()