"""
    Engineering Text-Length Features from a String Column
    Hard
    Data Transformation and Feature Engineering
    Write extract_features_from_text(records, column). Build a DataFrame and add two new columns derived from the text column column: length (character count, via .str.len()) and word_count (via .str.split().str.len()). Return the full table as a list of row dicts.

    Examples

    Arguments: [{'note': 'hello world'}, {'note': 'hi'}], 'note'
    Returns: [{'note': 'hello world', 'length': 11, 'word_count': 2}, {'note': 'hi', 'length': 2, 'word_count': 1}]
    Explanation: 'hello world' has length 11 and 2 words; 'hi' has length 2 and 1 word.
    Constraints

    records is a non-empty list of dicts sharing the same keys. column holds string values (possibly empty strings). 1 <= len(records) <= 500
"""

import pandas as pd

def extract_features_from_text(records, column):
    df = pd.DataFrame(records)
    df["length"] = df[column].str.len()
    df["word_count"] = df[column].str.split().str.len()
    return df.to_dict("records")
