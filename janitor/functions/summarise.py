"""Alternative function to pd.agg for summarizing data."""
from typing import Any

import pandas as pd
import pandas_flavor as pf

from janitor.functions.utils import _tupled, get_index_labels
from janitor.utils import check
from pandas.api.types import is_scalar
from typing import Hashable


@pf.register_dataframe_method
def summarise(
    df: pd.DataFrame,
    *args,
    by: Any = None,
) -> pd.DataFrame:
    """

    !!! info "New in version 0.25.0"

    !!!note

        Before reaching for `summarise`, try `pd.DataFrame.agg`.

    It is a wrapper around `pd.DataFrame.agg`,
    with added flexibility for multiple columns.

    It uses a variable argument of tuples, where the tuple is of
    the form `(col, func, name)`; `col` is the column
    label(s), `func` is the aggregation function,
    (which can be a supported Pandas string function, or a callable)
    or list of functions to be applied,
    while `name` is the new name to pass to the column
    (this is applicable only if `col` is a single column).
    `col` and `func` are required, `name` is optional.
    Column selection in `col` is possible using the
    [`select_columns`][janitor.functions.select.select_columns]
    syntax.

    Column selection in `by` is possible using the
    [`select_columns`][janitor.functions.select.select_columns]
    syntax.
    Arguments supported in `pd.DataFrame.groupby`
    can also be passed to `by` via a dictionary.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import janitor as jn
        >>> data = {'avg_jump': [3, 4, 1, 2, 3, 4],
        ...         'avg_run': [3, 4, 1, 3, 2, 4],
        ...         'avg_swim': [2, 1, 2, 2, 3, 4],
        ...         'combine_id': [100200, 100200,
        ...                        101200, 101200,
        ...                        102201, 103202],
        ...         'category': ['heats', 'heats',
        ...                      'finals', 'finals',
        ...                      'heats', 'finals']}
        >>> df = pd.DataFrame(data)
        >>> arg = col("avg_run").compute("mean")
        >>> df.summarize(arg, by=['combine_id', 'category'])
                                avg_run
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

        Summarize with a new column name:

        >>> arg = col("avg_run").compute("mean").rename("avg_run_2")
        >>> df.summarize(arg)
            avg_run_2
        0   2.833333
        >>> df.summarize(arg, by=['combine_id', 'category'])
                            avg_run_2
        combine_id category
        100200     heats         3.5
        101200     finals        2.0
        102201     heats         2.0
        103202     finals        4.0

        Summarize with the placeholders when renaming:

        >>> cols = col("avg*").compute("mean").rename("{_col}_{_fn}")
        >>> df.summarize(cols)
            avg_jump_mean  avg_run_mean  avg_swim_mean
        0       2.833333      2.833333       2.333333
        >>> df.summarize(cols, by=['combine_id', 'category'])
                                avg_jump_mean  avg_run_mean  avg_swim_mean
        combine_id category
        100200     heats               3.5           3.5            1.5
        101200     finals              1.5           2.0            2.0
        102201     heats               3.0           2.0            3.0
        103202     finals              4.0           4.0            4.0

        Pass the `col` class to `by`:

        >>> df.summarize(cols, by=col("c*"))
                                avg_jump_mean  avg_run_mean  avg_swim_mean
        combine_id category
        100200     heats               3.5           3.5            1.5
        101200     finals              1.5           2.0            2.0
        102201     heats               3.0           2.0            3.0
        103202     finals              4.0           4.0            4.0



    Args:
        df: A pandas DataFrame.
        args: variable arguments of tuple of the form (`col`, `func`, `name`).
        by: Column(s) to group by.

    Raises:
        ValueError: If a function is not provided.

    Returns:
        A pandas DataFrame with summarized columns.
    """  # noqa: E501

    if not args:
        raise ValueError("Kindly provide at least one aggregation tuple.")
    aggs = []
    for num, arg in enumerate(args):
        check("The summarise argument", arg, [tuple])
        if len(arg) < 2:
            raise ValueError(
                f"The tuple argument at position {num} "
                "should have at least a column "
                "and an aggregation function."
            )
        if len(arg) > 3:
            raise ValueError(
                f"The tuple argument at position {num} "
                "should be a maximum length of 3."
            )
        val = _tupled(*arg)
        func = val.func
        if isinstance(func, str) or callable(func):
            func = [func]
        check(
            f"func in tuple argument at position {num}",
            func,
            [str, callable, list],
        )
        if val.name:
            check(
                f"name in tuple argument at position {num}",
                val.name,
                [Hashable],
            )

        aggs.append(_tupled(col=val.col, func=func, name=val.name))
    aggs_ = []
    by_is_true = False
    if by:
        by_is_true = True
        if isinstance(by, dict):
            val = df.groupby(**by)
        else:
            by = get_index_labels(by, df, "columns")
            by = [by] if is_scalar(by) else by.tolist()
            val = df.groupby(list(by))
    else:
        val = df
    for ag in aggs:
        out = get_index_labels(ag.col, df, "columns")
        out = [out] if is_scalar(out) else out.tolist()
        if ("describe" in ag.func) and (len(ag.func) > 1):
            raise ValueError(
                "Kindly pass `describe` as a single func in a separate tuple"
            )
        if ag.func[0] == "describe":
            out = val[out].agg(ag.func[0])
        else:
            try:
                out = val[out].agg(ag.func)
            except AttributeError:
                if all(callable(func) for func in ag.func):
                    out = [val[out].pipe(func) for func in ag.func]
                    if len(out) == 1:
                        out = out[0]
        if (
            ag.name
            and isinstance(out, pd.DataFrame)
            and (len(out.columns) == 1)
        ):
            out.columns = pd.Index([ag.name])
        if isinstance(out, list):
            aggs_.extend(out)
        else:
            aggs_.append(out)
    if len(aggs_) > 1:
        no_of_levels = {frame.columns.nlevels for frame in aggs_}
        if len(no_of_levels) > 1:
            aggs = []
            len_no_of_levels = len(no_of_levels)
            for frame in aggs_:
                if frame.columns.nlevels < len_no_of_levels:
                    frame_nlevels = frame.columns.nlevels
                    size = len_no_of_levels - frame_nlevels
                    extra = [[""] * frame.columns.size] * size
                    columns = [
                        frame.columns.get_level_values(n)
                        for n in range(frame_nlevels)
                    ]
                    columns.extend(extra)
                    frame.columns = pd.MultiIndex.from_arrays(columns)
                aggs.append(frame)
            aggs_ = aggs
        if by_is_true:
            return pd.concat(aggs_, axis=1)
        return pd.concat(aggs_, axis=0)
    return aggs_[0]
