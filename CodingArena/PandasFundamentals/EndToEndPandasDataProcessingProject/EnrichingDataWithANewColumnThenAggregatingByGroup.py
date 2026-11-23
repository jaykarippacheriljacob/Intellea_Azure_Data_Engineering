"""
    Enriching Data with a New Column Then Aggregating by Group
    Medium
    End-to-End Pandas Data Processing Project
    Write enrich_and_group(records, group_col, value_col, multiplier). First
    add a new scaled column equal to value_col * multiplier, then return the
    SUM of scaled for each distinct value of group_col, as a
    {group_value: total} dict.

    Examples

    Arguments: [{'category': 'A', 'price': 10}, {'category': 'A', 'price': 20},
                {'category': 'B', 'price': 5}], 'category', 'price', 2
    Returns: {'A': 60, 'B': 10}
    Explanation: Every price is doubled first (20, 40, 10), THEN grouped:
    A totals 60, B totals 10.
    Constraints

    records is a non-empty list of dicts sharing the same keys. value_col
    holds numeric values. 1 <= len(records) <= 1000
"""
import pandas as pd


def enrich_and_group(records, group_col, value_col, multiplier):
    """
    new column scaled = value_col * multiplier -> groupby sum -> dict.

    Time:  O(n) - one pass to multiply, one to group and sum.
    Space: O(n) - for the DataFrame and the new column.
    """
    # Step 0: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 1: ENRICH - add a computed column. The multiply is vectorised:
    # every row of value_col is multiplied at once, no loop needed.
    #     price  scaled
    #     10     20
    #     20     40
    #      5     10
    df["scaled"] = df[value_col] * multiplier

    # Step 2: AGGREGATE - sum scaled per group. Result is a SERIES:
    #     category
    #     A    60
    #     B    10
    df1 = df.groupby(group_col)["scaled"].sum()

    # Step 3: Series -> {index: value} dict (no "records" - that's for a
    # DataFrame only).
    return df1.to_dict()


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Fully chained version (one expression):
#        return (pd.DataFrame(records)
#                  .assign(scaled=lambda d: d[value_col] * multiplier)
#                  .groupby(group_col)["scaled"].sum()
#                  .to_dict())
#    assign() adds a column and returns a NEW DataFrame, so it fits a chain.
#
# 2. Does the order (scale then sum) matter?
#      For multiplying: no - sum(x * k) == sum(x) * k, so
#      df.groupby(group_col)[value_col].sum() * multiplier gives the same.
#      But the task says "first add a new scaled column", and for
#      non-linear changes (e.g. squaring, rounding each row) the order DOES
#      matter. Follow the task's order.
#
# 3. Types:
#      int * int -> int  (60)        int * 1.5 -> float (45.0)
#      to_dict() on a Series returns plain Python int / float.
#
# 4. Adding a column - two ways:
#        df["scaled"] = df[value_col] * multiplier   (changes df in place)
#        df = df.assign(scaled=df[value_col] * multiplier)  (new DataFrame)
#
# 5. Common mistakes:
#      - df.scaled = ...               -> doesn't create a column (attribute)
#      - grouping on value_col         -> groups by price, wrong keys
#      - to_dict("records") on Series  -> TypeError
#      - summing value_col, not scaled -> forgets the multiplier
#
# 6. Edge cases worth naming in an interview:
#      - multiplier 0             -> every total 0
#      - negative multiplier      -> totals flip sign
#      - float multiplier         -> float totals
#      - one group                -> one key
#
# 7. Related: add_computed_column (step 1) and group_sum (step 2) - this
#    problem chains both: enrich -> aggregate.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'category': 'A', 'price': 10}, {'category': 'A', 'price': 20},
               {'category': 'B', 'price': 5}]
    big = [{'g': f"G{i % 5}", 'v': i} for i in range(1000)]
    big_expected = {f"G{k}": sum(range(k, 1000, 5)) * 3 for k in range(5)}

    tests = [
        # (records, group_col, value_col, multiplier, expected, description)
        (example, 'category', 'price', 2, {'A': 60, 'B': 10},
         "given example"),
        (example, 'category', 'price', 0, {'A': 0, 'B': 0}, "multiplier 0"),
        (example, 'category', 'price', -1, {'A': -30, 'B': -5},
         "negative multiplier"),
        (example, 'category', 'price', 1.5, {'A': 45.0, 'B': 7.5},
         "float multiplier"),
        ([{'g': 'x', 'v': 4}], 'g', 'v', 10, {'x': 40}, "single row"),
        (big, 'g', 'v', 3, big_expected, "max size n=1000"),
    ]

    passed = 0
    for records, group_col, value_col, multiplier, expected, desc in tests:
        got = enrich_and_group(records, group_col, value_col, multiplier)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"{{{len(got)} groups}}"
        print(f"{status}: {desc:<22} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = enrich_and_group(example, 'category', 'price', 2)
    assert type(out) is dict and type(out['A']) is int, "plain dict of int"
    assert 'scaled' not in example[0] and len(example) == 3, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()