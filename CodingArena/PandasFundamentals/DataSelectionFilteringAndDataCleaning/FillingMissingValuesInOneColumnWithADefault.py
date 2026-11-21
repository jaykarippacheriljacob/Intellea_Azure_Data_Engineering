"""
    Filling Missing Values in One Column with a Default
    Medium
    Data Selection, Filtering and Data Cleaning
    Write fill_missing_with_default(records, column, default). Build a DataFrame and replace every missing (None) value in column with default, using df[column].fillna(default). Return the full updated table as a list of row dicts.

    Examples

    Arguments: [{'name': 'Ravi', 'marks': 88}, {'name': 'Sam', 'marks': None}], 'marks', 0
    Returns: [{'name': 'Ravi', 'marks': 88.0}, {'name': 'Sam', 'marks': 0.0}]
    Explanation: Sam's missing marks becomes 0, Ravi's untouched 88 stays exactly as it was.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column may contain None values. 1 <= len(records) <= 500
"""

import pandas as pd

def fill_missing_with_default(records, column, default):
    df = pd.DataFrame(records)
    df[column] = df[column].fillna(default)
    return df.to_dict("records")
