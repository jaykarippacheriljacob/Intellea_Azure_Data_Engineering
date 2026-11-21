"""
    Summing a Column Grouped by Category
    Easy
    GroupBy, Aggregation, Merge and Concat
    Write group_sum(records, group_col, value_col). Build a DataFrame and return the total of value_col for each distinct value of group_col, using df.groupby(group_col)[value_col].sum(), as a {group_value: total} dict.

    Examples

    Arguments: [{'category': 'A', 'amount': 10}, {'category': 'B', 'amount': 5}, {'category': 'A', 'amount': 20}], 'category', 'amount'
    Returns: {'A': 30, 'B': 5}
    Explanation: A's two amounts (10 + 20) sum to 30; B has just one, 5.
    Constraints

    records is a non-empty list of dicts sharing the same keys. value_col holds numeric values. 1 <= len(records) <= 1000
"""

