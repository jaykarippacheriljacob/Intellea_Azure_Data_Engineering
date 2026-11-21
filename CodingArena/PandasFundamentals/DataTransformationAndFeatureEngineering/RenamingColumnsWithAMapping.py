"""
    Renaming Columns with a Mapping
    Easy
    Data Transformation and Feature Engineering
    Write rename_columns(records, mapping). Build a DataFrame and rename its
    columns according to the {old_name: new_name} dict mapping, using
    df.rename(columns=mapping). Columns not mentioned in mapping keep their
    original name.

    Examples

    Arguments: [{'marks': 88, 'age': 23}], {'marks': 'score'}
    Returns: [{'score': 88, 'age': 23}]
    Explanation: marks becomes score; age, not mentioned in the mapping,
    keeps its original name.
    Constraints

    records is a non-empty list of dicts sharing the same keys. Every key in
    mapping is an actual column. 1 <= len(records) <= 500
"""
import pandas as pd


def rename_columns(records, mapping):
    """
    Rename columns with {old: new}; unmentioned columns keep their names.
    Return the table as row dicts.

    Time:  O(n * k) - building the DataFrame and converting back.
    Space: O(n * k) - for the DataFrame.
    """
    # Step 1: list of dicts -> DataFrame.
    df = pd.DataFrame(records)

    # Step 2: rename(columns={old: new}) returns a NEW DataFrame -> assign
    # it back. Columns missing from the mapping are left as they are.
    # The keyword `columns=` matters: rename(mapping) alone would try to
    # rename the INDEX (row labels), not the columns.
    df = df.rename(columns=mapping)

    # REVISION 1 (bug fix): the first version returned `df` itself - a
    # DataFrame. The example output is a list of row dicts, so convert.
    return df.to_dict("records")


# ---------------------------------------------------------------------------
# LeetCode-style notes / possible follow-ups
# ---------------------------------------------------------------------------
# 1. Pure-Python version (no pandas):
#        return [{mapping.get(k, k): v for k, v in r.items()} for r in records]
#    mapping.get(k, k) -> the new name if there is one, else the old name.
#
# 2. Other ways to rename:
#        df.columns = ["score", "age"]       replace ALL names (must list
#                                            every column, in order)
#        df.rename(columns=str.upper)        apply a function to every name
#        df.columns = df.columns.str.lower().str.replace(" ", "_")
#                                            clean messy headers
#        df.rename(index={0: "first"})       rename ROW labels instead
#
# 3. Common mistakes:
#      - returning df instead of df.to_dict("records")
#      - df.rename(columns=mapping) without "df =" -> no change
#      - df.rename(mapping) without columns= -> renames the index, columns
#        stay the same (and no error is raised!)
#      - mapping written backwards {new: old} -> nothing matches, no error
#
# 4. Edge cases worth naming in an interview:
#      - empty mapping {}             -> table unchanged
#      - rename every column          -> all new names
#      - swap two names {a: b, b: a}  -> works, rename is applied at once
#      - column order                 -> unchanged, only names change
#      - values                       -> unchanged
#
# 5. Related problems: LeetCode 2885 (Rename Columns).


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    big = [{'a': i, 'b': i * 2} for i in range(500)]

    tests = [
        # (records, mapping, expected, description)
        ([{'marks': 88, 'age': 23}], {'marks': 'score'},
         [{'score': 88, 'age': 23}], "given example"),
        ([{'a': 1, 'b': 2}], {}, [{'a': 1, 'b': 2}], "empty mapping"),
        ([{'a': 1, 'b': 2}], {'a': 'x', 'b': 'y'}, [{'x': 1, 'y': 2}],
         "rename every column"),
        ([{'a': 1, 'b': 2}], {'a': 'b', 'b': 'a'}, [{'b': 1, 'a': 2}],
         "swap two names"),
        ([{'c': 1, 'a': 2, 'b': 3}], {'a': 'z'}, [{'c': 1, 'z': 2, 'b': 3}],
         "column order kept"),
        ([{'n': 'Ravi', 'm': 88}, {'n': 'Sam', 'm': 91}], {'n': 'name'},
         [{'name': 'Ravi', 'm': 88}, {'name': 'Sam', 'm': 91}], "many rows"),
        (big, {'b': 'double'}, [{'a': i, 'double': i * 2} for i in range(500)],
         "max size n=500"),
    ]

    passed = 0
    for records, mapping, expected, desc in tests:
        got = rename_columns(records, mapping)
        ok = got == expected and all(list(g) == list(e) for g, e in zip(got, expected))
        passed += ok
        status = "PASS" if ok else "FAIL"
        shown = got if len(got) <= 2 else f"[{len(got)} rows]"
        print(f"{status}: {desc:<20} -> {shown}")
        if not ok:
            print(f"      expected {expected}")

    # Extra checks on behaviour, not just values
    data = [{'marks': 88, 'age': 23}]
    out = rename_columns(data, {'marks': 'score'})
    assert type(out) is list and type(out[0]) is dict, "list of row dicts"
    assert data == [{'marks': 88, 'age': 23}], "input unchanged"
    # the trap from note 3: without columns= nothing happens to the columns
    assert list(pd.DataFrame(data).rename({'marks': 'score'}).columns) == ['marks', 'age']
    print("PASS: return type / input untouched / rename(mapping) trap")

    print(f"\n{passed}/{len(tests)} value tests passed")


if __name__ == "__main__":
    run_tests()