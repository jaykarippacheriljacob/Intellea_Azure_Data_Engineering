"""
    Building a Full Clean-Transform-Aggregate Report
    Hard
    End-to-End Pandas Data Processing Project
    Write end_to_end_report(records) - a complete mini pipeline over rows
    shaped like {"category": ..., "amount": ...}. Drop rows with any missing
    value, drop duplicate rows, then group by category and compute the sum
    and count of amount for each. Return {category: {"sum": ...,
    "count": ...}}. If NOTHING survives cleaning, return an empty dict.

    Examples

    Arguments: [{'category': 'A', 'amount': 10},
                {'category': 'A', 'amount': 20},
                {'category': 'B', 'amount': 5}]
    Returns: {'A': {'sum': 30, 'count': 2}, 'B': {'sum': 5, 'count': 1}}
    Explanation: Category A's two amounts sum to 30 across 2 rows; category B
    has 1 row summing to 5.
    Constraints

    records is a non-empty list of dicts, each with a category key (string)
    and an amount key (number or None). 1 <= len(records) <= 1000
"""
import pandas as pd


def end_to_end_report(records):
    """
    dropna -> drop_duplicates -> groupby sum + count -> to_dict("index").

    Time:  O(n) - each cleaning step and the groupby are one pass.
    Space: O(n) - for the DataFrame.
    """
    # Step 0: list of dicts -> DataFrame. None -> NaN; if amount has a NaN
    # the column becomes float64, so sums come back as e.g. 10.0.
    df = pd.DataFrame(records)

    # Step 1: CLEAN - drop rows with a missing value in ANY column
    # (task says "any missing value" -> plain dropna, no subset).
    df = df.dropna()

    # Step 2: CLEAN - drop repeated rows (keep the first).
    # Order: dropna first, then drop_duplicates, as the task lists.
    df = df.drop_duplicates()

    # Step 3: AGGREGATE - two stats per category in one call. A list in
    # SQUARE brackets gives a DataFrame: index = categories, one column per
    # stat.
    #              sum  count
    #     category
    #     A         30      2
    #     B          5      1
    df1 = df.groupby("category")["amount"].agg(["sum", "count"])

    # Step 4: {category: {stat: value}} -> orient "index" (index = outer
    # keys). Empty case needs no check: an empty groupby -> empty DataFrame
    # -> to_dict("index") gives {}.
    return df1.to_dict("index")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Fully chained version (one expression):
#        return (pd.DataFrame(records)
#                  .dropna()
#                  .drop_duplicates()
#                  .groupby("category")["amount"].agg(["sum", "count"])
#                  .to_dict("index"))
#
# 2. Why the cleaning order matters for the COUNT:
#      Without drop_duplicates, two identical rows (A, 10) count as 2 and
#      sum to 20. Cleaning BEFORE grouping is what makes the report right.
#      Standard workflow: clean -> transform -> aggregate -> report.
#
# 3. to_dict orients (pick by the expected output's OUTER keys):
#        "index"   -> {category: {stat: val}}   <- here
#        default   -> {stat: {category: val}}   flipped
#        "records" -> [{stat: val}, ...]         categories lost
#
# 4. sum vs count types:
#        count -> always int (number of rows)
#        sum   -> int if amount had no None, float if it did (NaN made the
#                 column float before dropna removed the row)
#
# 5. Why no empty check (unlike clean_and_summarize):
#      groupby on an empty table gives no groups -> {} directly, which is
#      exactly what the task wants. mean() on an empty Series would give NaN
#      - sum/count via groupby don't have that problem.
#
# 6. Common mistakes:
#      - forgetting drop_duplicates   -> repeated rows inflate sum and count
#      - to_dict("records")           -> list, category names lost
#      - agg("sum", "count")          -> needs a LIST: agg(["sum", "count"])
#      - dropna(subset=["amount"])    -> keeps rows with a missing category
#
# 7. Edge cases worth naming in an interview:
#      - every amount None            -> {}
#      - missing category             -> row dropped (any column)
#      - exact duplicate rows         -> counted once
#      - same category, different amounts -> NOT duplicates, both kept
#
# 8. Related: filter_and_clean (dropna -> drop_duplicates) and
#    group_multiple_agg (agg list -> to_dict("index")) - this problem
#    chains both into one report.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    example = [{'category': 'A', 'amount': 10},
               {'category': 'A', 'amount': 20},
               {'category': 'B', 'amount': 5}]
    big = [{'category': f"C{i % 3}", 'amount': None if i % 50 == 0 else i % 40}
           for i in range(1000)]
    big_expected = (pd.DataFrame(big).dropna().drop_duplicates()
                      .groupby("category")["amount"].agg(["sum", "count"])
                      .to_dict("index"))

    tests = [
        # (records, expected, description)
        (example, {'A': {'sum': 30, 'count': 2}, 'B': {'sum': 5, 'count': 1}},
         "given example"),
        ([{'category': 'A', 'amount': None}], {}, "nothing survives"),
        ([{'category': 'A', 'amount': 10}, {'category': 'A', 'amount': 10}],
         {'A': {'sum': 10, 'count': 1}}, "duplicate collapses"),
        ([{'category': 'A', 'amount': 10}, {'category': 'B', 'amount': None},
          {'category': None, 'amount': 3}],
         {'A': {'sum': 10.0, 'count': 1}}, "missing in either column"),
        ([{'category': 'A', 'amount': 7}, {'category': 'A', 'amount': 3}],
         {'A': {'sum': 10, 'count': 2}}, "same category, not duplicate"),
        (big, big_expected, "max size n=1000"),
    ]

    passed = 0
    for records, expected, desc in tests:
        got = end_to_end_report(records)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 1 else f"{{{len(got)} categories}}"
        print(f"{status}: {desc:<30} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = end_to_end_report(example)
    assert type(out) is dict and type(out['A']['count']) is int, "plain types"
    assert len(example) == 3 and example[0]['amount'] == 10, "input unchanged"
    print("PASS: return type / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()