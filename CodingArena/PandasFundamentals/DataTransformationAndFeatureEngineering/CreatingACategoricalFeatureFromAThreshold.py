"""
    Creating a Categorical Feature from a Threshold
    Medium
    Data Transformation and Feature Engineering
    Write bucket_by_condition(records, column, threshold, low_label,
    high_label). Build a DataFrame and add a new category column: high_label
    where column is strictly greater than threshold, otherwise low_label -
    using np.where. Return the full table as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 40}],
               'marks', 50, 'Fail', 'Pass'
    Returns: [{'name': 'Ravi', 'marks': 88, 'category': 'Pass'},
              {'name': 'Sam', 'marks': 40, 'category': 'Fail'}]
    Explanation: Ravi's 88 clears 50 and gets 'Pass'; Sam's 40 gets 'Fail'.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds
    numeric values. 1 <= len(records) <= 500
"""
import numpy as np
import pandas as pd


def bucket_by_condition(records, column, threshold, low_label, high_label):
    """
    Add `category` = high_label where column > threshold, else low_label.
    Return the full table as row dicts.

    Time:  O(n * k) - building the DataFrame; np.where is one O(n) pass.
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 2: np.where(condition, value_if_true, value_if_false)
    #   df[column] > threshold -> True/False for every row (a mask)
    #   True  -> high_label,  False -> low_label
    # Works on the whole column at once - no loop.
    #
    # REVISION 1: the first version built the list with a for-loop + if/else.
    # It gives the same result, but the task asks for np.where, and np was
    # imported but never used. The loop also runs row by row in Python,
    # which is slower on big tables.
    df["category"] = np.where(df[column] > threshold, high_label, low_label)

    # Step 3: full table -> row dicts (labels come back as plain str).
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. The first (loop) version - correct output, but not what was asked:
#        ls = []
#        for val in df[column]:
#            if val > threshold:
#                ls.append(high_label)
#            else:
#                ls.append(low_label)
#        df["category"] = ls
#
# 2. Watch the ARGUMENT ORDER: the function takes (low_label, high_label),
#    but np.where takes (condition, if_TRUE, if_FALSE). The condition is
#    "> threshold", so if_TRUE = high_label. Swapping them is the most
#    likely bug in this problem.
#
# 3. Other ways to do the same thing:
#        df[column].apply(lambda v: high_label if v > threshold else low_label)
#        df["category"] = low_label
#        df.loc[df[column] > threshold, "category"] = high_label
#
# 4. More than two buckets:
#        np.select([cond1, cond2], ["A", "B"], default="C")
#        pd.cut(df[col], bins=[0, 50, 75, 100], labels=["Low", "Mid", "High"])
#
# 5. Edge cases worth naming in an interview:
#      - value equal to threshold -> low_label (strictly greater)
#      - all above / all below    -> one label everywhere
#      - negative / float values  -> work as normal
#      - NaN (not allowed here)   -> NaN > x is False -> low_label
#
# 6. Related: NumPy Fundamentals - np.where for conditional replacement;
#    filter_by_condition uses the same "> threshold" mask.


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'id': i, 'v': i} for i in range(500)]

    tests = [
        # (records, column, threshold, low, high, expected, description)
        ([{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': 40}],
         'marks', 50, 'Fail', 'Pass',
         [{'name': 'Ravi', 'marks': 88, 'category': 'Pass'},
          {'name': 'Sam', 'marks': 40, 'category': 'Fail'}], "given example"),
        ([{'m': 50}], 'm', 50, 'Low', 'High',
         [{'m': 50, 'category': 'Low'}], "equal -> low_label"),
        ([{'m': 60}, {'m': 70}], 'm', 50, 'Low', 'High',
         [{'m': 60, 'category': 'High'}, {'m': 70, 'category': 'High'}],
         "all above"),
        ([{'m': 1}, {'m': 2}], 'm', 50, 'Low', 'High',
         [{'m': 1, 'category': 'Low'}, {'m': 2, 'category': 'Low'}],
         "all below"),
        ([{'t': -5}, {'t': 3}], 't', 0, 'Cold', 'Warm',
         [{'t': -5, 'category': 'Cold'}, {'t': 3, 'category': 'Warm'}],
         "negative values"),
        ([{'p': 2.5}, {'p': 2.4}], 'p', 2.45, 'No', 'Yes',
         [{'p': 2.5, 'category': 'Yes'}, {'p': 2.4, 'category': 'No'}],
         "float threshold"),
        (big, 'v', 249, 'L', 'H',
         [{'id': i, 'v': i, 'category': 'H' if i > 249 else 'L'}
          for i in range(500)], "max size n=500"),
    ]

    passed = 0
    for records, column, th, low, high, expected, desc in tests:
        got = bucket_by_condition(records, column, th, low, high)
        ok = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<20} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [{'name': 'Ravi', 'marks': 88}]
    out = bucket_by_condition(data, 'marks', 50, 'Fail', 'Pass')
    assert type(out[0]['category']) is str, "plain Python str"
    assert list(out[0]) == ['name', 'marks', 'category'], "new column last"
    assert 'category' not in data[0], "input unchanged"
    print("PASS: str type / column order / input untouched")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()