"""
    Classifying Each Column as Numeric or Text
    Hard
    Introduction to Pandas and Data Exploration
    Write dtype_kind_report(records). Build a DataFrame and, for every column,
    report whether pandas inferred a NUMERIC dtype or a TEXT dtype, using
    pd.api.types.is_numeric_dtype. Return {column: "numeric" | "text"}.

    Examples

    Arguments: [{'name': 'Ravi', 'age': 23}, {'name': 'Sam', 'age': 25}]
    Returns: {'name': 'text', 'age': 'numeric'}
    Explanation: name is text, age is numeric:
    {'name': 'text', 'age': 'numeric'}.
    Constraints

    records is a non-empty list of dicts sharing the same keys, where every
    column's values are consistently either all numbers or all strings.
    1 <= len(records) <= 500
"""
import pandas as pd


def dtype_kind_report(records):
    """
    Return {column: "numeric" | "text"} for every column, in column order.

    Time:  O(n * k) - building the DataFrame; each dtype check is O(1).
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame. pandas INFERS each column's dtype:
    # numbers -> int64 / float64, text -> str (object in older pandas).
    df = pd.DataFrame(records)

    # Step 2: check each column ON ITS OWN.
    # FIRST ATTEMPT (wrong): is_numeric_dtype(df) on the WHOLE DataFrame
    # returns ONE bool (False), not one answer per column. Then
    # False.to_dict("records") -> AttributeError: 'bool' object has no
    # attribute 'to_dict'.
    df_type = {}
    for col in df.columns:
        # df[col] is one Series -> one True/False for that column.
        # `== True` is not needed: is_numeric_dtype already returns a bool.
        if pd.api.types.is_numeric_dtype(df[col]):
            df_type[col] = "numeric"
        else:
            df_type[col] = "text"

    # Step 3: return the dict (the grader reads the RETURN value, not print).
    return df_type


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Dict-comprehension version (same result):
#        return {col: "numeric" if pd.api.types.is_numeric_dtype(df[col])
#                     else "text"
#                for col in df.columns}
#
# 2. Other ways to get the same information:
#        df.dtypes                               dtype of every column
#        df.select_dtypes(include="number").columns   only numeric columns
#        df.select_dtypes(exclude="number").columns   everything else
#
# 3. The checker takes a dtype OR a Series OR an array:
#        is_numeric_dtype(df["age"])        -> True
#        is_numeric_dtype(df["age"].dtype)  -> True
#        is_numeric_dtype(df)               -> False (one answer, no loop)
#
# 4. Results for different column contents (tested):
#        [1.5]        float64            -> numeric
#        [1, None]    float64 (NaN)      -> numeric
#        ['5']        str                -> text  (a number written as text)
#        [None]       object             -> text
#        [True]       bool               -> numeric (pandas counts bool as
#                                           numeric - known interview trap)
#
# 5. Edge cases worth naming in an interview:
#      - all columns numeric / all text  -> every value the same label
#      - mixed int and float in a column -> float64 -> numeric
#      - numbers stored as strings       -> text; convert with
#        pd.to_numeric(df[col]) or .astype(int) first if needed
#      - column order follows the keys of the dicts
#
# 6. Related: df.dtypes and df.info() (Pandas Fundamentals 1.5 / 1.6),
#    astype() and pd.to_datetime() (2.9).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'id': i, 'name': f"n{i}", 'score': i * 1.5} for i in range(500)]

    tests = [
        # (records, expected, description)
        ([{'name': 'Ravi', 'age': 23}, {'name': 'Sam', 'age': 25}],
         {'name': 'text', 'age': 'numeric'}, "given example"),
        ([{'a': 1, 'b': 2.5}], {'a': 'numeric', 'b': 'numeric'},
         "all numeric (int + float)"),
        ([{'a': 'x', 'b': 'y'}], {'a': 'text', 'b': 'text'}, "all text"),
        ([{'v': 1}, {'v': 2.5}], {'v': 'numeric'}, "int and float mixed"),
        ([{'code': '5'}, {'code': '7'}], {'code': 'text'}, "numbers as strings"),
        ([{'z': 1, 'y': 'a', 'x': 2}], {'z': 'numeric', 'y': 'text', 'x': 'numeric'},
         "column order kept"),
        (big, {'id': 'numeric', 'name': 'text', 'score': 'numeric'},
         "max size n=500"),
    ]

    passed = 0
    for records, expected, desc in tests:
        got = dtype_kind_report(records)
        ok = got == expected and list(got) == list(expected)
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {desc:<26} -> {got}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    out = dtype_kind_report([{'name': 'Ravi', 'age': 23}])
    assert type(out) is dict, "returns a dict"
    assert set(out.values()) <= {"numeric", "text"}, "only the two labels"
    print("PASS: return type / labels")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()