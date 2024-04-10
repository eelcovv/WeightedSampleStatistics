import re


def make_negation_name(column_name, suffix="_x") -> str:
    """Make a new column name based for the negative value

    Returns
    -------
    new_col : str
    """
    match = re.search("(_\d\.\d)$", column_name)
    if match:
        # column ends with _1.0, make the new name like _x_1.0
        new_col = "".join([re.sub("_\d\.\d$", "", column_name), suffix]) + match.group(
            1
        )
    else:
        new_col = "".join([column_name, suffix])
    return new_col
