"""
    Concatenating Two Tables of Rows
    Easy
    GroupBy, Aggregation, Merge and Concat
    Write concat_two_tables(records_a, records_b). Stack the rows of
    records_b underneath records_a using pd.concat, returning the combined
    table as a list of row dicts, with records_a's rows first.

    Examples

    Arguments: [{'name': 'Ravi'}], [{'name': 'Sam'}, {'name': 'Ana'}]
    Returns: [{'name': 'Ravi'}, {'name': 'Sam'}, {'name': 'Ana'}]
    Explanation: Ravi's row comes first, followed by Sam's and Ana's, in that
    order.
    Constraints

    records_a and records_b are lists of dicts sharing the same keys as each
    other (when both are non-empty). Either list may be empty.
    0 <= len(records_a) + len(records_b) <= 1000
"""
import pandas as pd


def concat_two_tables(records_a, records_b):
    """
    pd.concat([A, B]) -> rows of B under rows of A -> list of row dicts.

    Time:  O(n + m) - copies every row of both tables once.
    Space: O(n + m) - for the DataFrames and the result.
    """
    # Step 0: both lists of dicts -> DataFrames (an empty list gives an
    # empty DataFrame with no columns - concat handles that fine).
    df1 = pd.DataFrame(records_a)
    df2 = pd.DataFrame(records_b)

    # Step 1: stack vertically. The ORDER in the list = order of rows,
    # so [df1, df2] puts A's rows first.
    # ignore_index=True renumbers rows 0, 1, 2 ... instead of keeping the
    # old labels (0, 0, 1). The output dicts don't show the index, but a
    # clean index avoids surprises if you use loc later.
    df3 = pd.concat([df1, df2], ignore_index=True)

    # Step 2: DataFrame -> list of row dicts ("records" is a keyword).
    return df3.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. concat vs merge:
#        pd.concat  -> STACKS tables (rows under rows), no key matching
#        pd.merge   -> JOINS tables side by side on a shared key
#    concat = "add more rows"; merge = "add more columns by matching".
#
# 2. pd.concat takes a LIST:
#        pd.concat([df1, df2])     correct
#        pd.concat(df1, df2)       TypeError (second arg is not a table)
#
# 3. axis:
#        axis=0 (default) -> stack rows (this problem)
#        axis=1           -> put tables side by side by index position
#
# 4. Different columns in the two tables:
#      concat keeps ALL columns; a row missing a column gets NaN there
#      (and an int column with NaN becomes float).
#      join="inner" keeps only the columns both tables share.
#
# 5. Without ignore_index=True:
#        index becomes 0, 0, 1  (old labels kept) -> df.loc[0] returns
#        TWO rows. Harmless for to_dict("records"), risky elsewhere.
#
# 6. Real use: Midstone 2 daily files - orders day=01, day=02 ... are
#    stacked with pd.concat([...], ignore_index=True) into one table.
#
# 7. Edge cases worth naming in an interview:
#      - A empty                -> only B's rows
#      - B empty                -> only A's rows
#      - both empty             -> []
#      - duplicate rows across  -> kept (concat doesn't remove; use
#        drop_duplicates() after if needed)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big_a = [{'k': i} for i in range(600)]
    big_b = [{'k': i} for i in range(600, 1000)]

    tests = [
        # (records_a, records_b, expected, description)
        ([{'name': 'Ravi'}], [{'name': 'Sam'}, {'name': 'Ana'}],
         [{'name': 'Ravi'}, {'name': 'Sam'}, {'name': 'Ana'}],
         "given example"),
        ([], [{'n': 1}], [{'n': 1}], "A empty"),
        ([{'n': 1}], [], [{'n': 1}], "B empty"),
        ([], [], [], "both empty"),
        ([{'n': 1}], [{'n': 1}], [{'n': 1}, {'n': 1}], "duplicates kept"),
        ([{'a': 1, 'b': 'x'}], [{'a': 2, 'b': 'y'}],
         [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], "several columns"),
        (big_a, big_b, [{'k': i} for i in range(1000)], "max size n=1000"),
    ]

    passed = 0
    for records_a, records_b, expected, desc in tests:
        got = concat_two_tables(records_a, records_b)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<20} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = concat_two_tables([], [{'n': 1}])
    assert type(out) is list and type(out[0]['n']) is int, "int kept"
    a = [{'name': 'Ravi'}]
    concat_two_tables(a, [{'name': 'Sam'}])
    assert a == [{'name': 'Ravi'}], "input unchanged"
    print("PASS: return type / int kept / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()