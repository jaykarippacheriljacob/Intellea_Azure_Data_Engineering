"""
    Adding a New Computed Column
    Easy
    Data Transformation and Feature Engineering
    Write add_computed_column(records, column, multiplier). Build a DataFrame
    and add a new column, scaled_value, equal to column * multiplier for every
    row. Return the full table (original columns plus the new one) as a list
    of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'price': 100}, {'name': 'Sam', 'price': 200}],
               'price', 2
    Returns: [{'name': 'Ravi', 'price': 100, 'scaled_value': 200},
              {'name': 'Sam', 'price': 200, 'scaled_value': 400}]
    Explanation: Every row's price is doubled into a new scaled_value column:
    200 and 400.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds
    numeric values. 1 <= len(records) <= 500
"""
import pandas as pd


def add_computed_column(records, column, multiplier):
    """
    Return every row of `records` plus a new key `scaled_value`
    = row[column] * multiplier, as a list of row dicts.

    Time:  O(n * k) - n rows, k columns (build, multiply, convert back).
    Space: O(n * k) - for the DataFrame and the output list.
    """
    # Step 1: list of dicts -> DataFrame (each dict is one row).
    df = pd.DataFrame(records)

    # Step 2: vectorised arithmetic. df[column] * multiplier multiplies the
    # WHOLE column at once (no loop). Assigning to a name that doesn't exist
    # yet, df["scaled_value"], creates a new column at the END of the table.
    df["scaled_value"] = df[column] * multiplier

    # Step 3: DataFrame -> list of row dicts.
    # "records" is a FIXED KEYWORD of to_dict (one dict per row), not the
    # function's variable `records`.
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas):
#        def add_computed_column(records, column, multiplier):
#            return [{**row, "scaled_value": row[column] * multiplier}
#                    for row in records]
#    {**row, ...} copies the row and adds a key, so the input is untouched.
#
# 2. Other pandas ways to add a column:
#        df = df.assign(scaled_value=df[column] * multiplier)  # returns a NEW df
#        df.insert(1, "scaled_value", df[column] * multiplier) # at a position
#        df["scaled_value"] = df[column].apply(lambda x: x * multiplier)
#                                     # works, but slower: avoid for maths
#
# 3. Creating vs overwriting:
#        df["new"] = ...    -> creates the column if the name is new
#        df["price"] = ...  -> OVERWRITES the existing column
#    df.new = ... does NOT create a column (sets an attribute) - always use
#    square brackets to create one.
#
# 4. dtype of the new column:
#        int  * int   -> int    (100 * 2 = 200)
#        int  * float -> float  (100 * 1.5 = 150.0)
#
# 5. Edge cases worth naming in an interview:
#      - multiplier 0          -> all scaled_value = 0
#      - negative multiplier   -> negative values
#      - float multiplier      -> float column
#      - NaN in the column     -> NaN * x = NaN (no error)
#      - column order          -> new column is added LAST
#
# 6. Related problems: LeetCode 2881 (Create a New Column),
#    LeetCode 2884 (Modify Columns).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    items = [{'name': 'Ravi', 'price': 100}, {'name': 'Sam', 'price': 200}]
    big = [{'id': i, 'v': i} for i in range(500)]

    tests = [
        # (records, column, multiplier, expected, description)
        (items, 'price', 2,
         [{'name': 'Ravi', 'price': 100, 'scaled_value': 200},
          {'name': 'Sam', 'price': 200, 'scaled_value': 400}],
         "given example"),
        (items, 'price', 0,
         [{'name': 'Ravi', 'price': 100, 'scaled_value': 0},
          {'name': 'Sam', 'price': 200, 'scaled_value': 0}],
         "multiplier 0"),
        (items, 'price', -1,
         [{'name': 'Ravi', 'price': 100, 'scaled_value': -100},
          {'name': 'Sam', 'price': 200, 'scaled_value': -200}],
         "negative multiplier"),
        (items, 'price', 1.5,
         [{'name': 'Ravi', 'price': 100, 'scaled_value': 150.0},
          {'name': 'Sam', 'price': 200, 'scaled_value': 300.0}],
         "float multiplier"),
        ([{'x': 2.5}], 'x', 4,
         [{'x': 2.5, 'scaled_value': 10.0}],
         "single row, float column"),
        (big, 'v', 3,
         [{'id': i, 'v': i, 'scaled_value': i * 3} for i in range(500)],
         "max size n=500"),
    ]

    passed = 0
    for records, column, multiplier, expected, desc in tests:
        got = add_computed_column(records, column, multiplier)
        # also check the new column comes LAST
        ok = got == expected and all(list(r)[-1] == 'scaled_value' for r in got)
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<24} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = add_computed_column(items, 'price', 2)
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert type(out[0]['scaled_value']) is int, "plain Python int, not numpy"
    assert items[0] == {'name': 'Ravi', 'price': 100}, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()