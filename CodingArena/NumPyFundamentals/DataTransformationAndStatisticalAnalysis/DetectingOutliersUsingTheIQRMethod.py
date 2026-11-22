"""
    Detecting Outliers Using the IQR Method
    Hard
    Data Transformation & Statistical Analysis
    Write iqr_outliers(values). Compute the 25th and 75th percentiles of values, derive the interquartile range (IQR = Q3 - Q1), and return every value falling below Q1 - 1.5*IQR or above Q3 + 1.5*IQR as a plain list, in original order. With fewer than 4 values there isn't enough data to compute meaningful quartiles, so return an empty list instead.

    Examples

    Arguments: [1, 2, 2, 3, 2, 90]
    Returns: [90.0]
    Explanation: The single extreme value, 90, is far enough outside the typical spread to be flagged.
    Constraints

    values is a list of numbers. Fewer than 4 values always returns an empty list, regardless of content. 0 <= len(values) <= 1000
"""

