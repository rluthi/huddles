from typing import Union
import altair
import numpy as np
import pandas as pd


def create_exploration_plot(
    embeddings: Union[np.ndarray, pd.DataFrame],
    dataset: pd.DataFrame,
    hue: Union[str, None] = None,
    disable_row_limit: bool = True,
) -> altair.vegalite.v4.api.VConcatChart:
    """
    Visualize a 2D representation of your data you can interact with.

    The function creates an interactive scatter plot of 2D embeddings using Altair. It allows for optional color coding
    based on a feature specified by 'hue'. A data table corresponding to the points currently selected is also
    generated.

    Arguments:
        embeddings: A 2D NumPy array or Pandas DataFrame containing the embeddings. Must have exactly 2 columns.
        dataset: Pandas DataFrame corresponding to the embeddings. The number of rows should match the embeddings.
        hue: Optional; column name in 'dataset' for color coding. If None, a default color is used.
        disable_row_limit: Optional; if True, disables Altair's maximum row limit.

    Returns:
        An Altair vertical concatenation chart (scatter plot and data table).
    """

    # Sanity checks
    if embeddings.shape[1] != 2:
        raise ValueError(f"The embeddings has {embeddings.shape[1]}) columns. It must be 2! ")

    if len(dataset) != embeddings.shape[0]:
        raise ValueError(
            f"The number of rows ({len(dataset)}) must match the length of the embeddings array ({embeddings.shape[0]})"
        )

    if (hue is not None) and (hue not in dataset.columns):
        raise ValueError(f"hue ('{hue}') must be one of the dataset's columns: ({list(dataset.columns)})")

    # Handling when given a pd.DataFrame for the 'embeddings'
    if isinstance(embeddings, pd.DataFrame):
        embeddings = embeddings.values

    # Concatenating embeddings and data into a single dataframe
    dataset_cols = list(dataset.columns)
    df_ = pd.concat(
        [pd.DataFrame({"x1": embeddings[:, 0], "x2": embeddings[:, 1]}), dataset],
        axis=1,
    )

    # Setting altair's row limit
    if disable_row_limit:
        altair.data_transformers.disable_max_rows()

    # Plotting
    brush = altair.selection_interval()

    if hue is None:
        color = altair.condition(brush, altair.value("steelblue"), altair.value("grey"))
    else:
        color = altair.Color(f"{hue}")

    scatter_plot = (
        altair.Chart(df_)
        .mark_circle(opacity=0.6, size=20)
        .encode(
            x=altair.X("x1:Q", axis=None, scale=altair.Scale(zero=False)),
            y=altair.Y("x2:Q", axis=None, scale=altair.Scale(zero=False)),
            color=color,
            tooltip=dataset_cols,
        )
        .properties(width=350, height=350, title="2D Embedding Space")
        .add_params(brush)
    )

    # Base chart for data tables
    ranked_text = (  # more info: https://altair-viz.github.io/gallery/scatter_linked_table.html
        altair.Chart(df_)
        .mark_text(align="right")
        .encode(
            y=altair.Y("row_number:O", axis=None),
        )
        .transform_filter(brush)
        .transform_window(row_number="row_number()")
        .transform_window(rank="rank(row_number)")  # I do need these two lines for the right columns to render!
        .transform_filter(altair.datum.rank < 18)
        .properties(title="Dataset")
    )

    # Data Tables
    # sourcery skip: for-append-to-extend, list-comprehension
    alt_data_columns = []
    for column_name in dataset_cols:
        alt_data_columns.append(
            ranked_text.encode(text=column_name).properties(title=altair.Title(text=column_name, align="right"))
        )

    data_table = altair.hconcat(*alt_data_columns)  # Combine data tables

    return (scatter_plot | data_table).configure_axis(grid=False).configure_view(strokeWidth=0)
