import pandas as pd


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def smoothing_label(smooth_cases_window: int) -> str:
    window = int(smooth_cases_window)
    if window <= 0:
        return "raw daily cases"
    return f"causal {window}-day moving-average cases"


def matched_window_suffix(
    matched_window_with_opentable: bool,
    matched_source: str = "ot",
) -> str:
    if not matched_window_with_opentable:
        return ""
    return f"_matched_{matched_source}"


def online_artifact_stem(
    split: str,
    asof: str,
    window_days: int,
    smooth_cases_window: int,
    matched_window_with_opentable: bool = False,
    matched_source: str = "ot",
) -> str:
    suffix = matched_window_suffix(
        matched_window_with_opentable=matched_window_with_opentable,
        matched_source=matched_source,
    )
    if matched_window_with_opentable:
        return f"{split}_{asof}{suffix}_w{int(smooth_cases_window)}"
    return f"{split}_{asof}_history{int(window_days)}_w{int(smooth_cases_window)}"


def private_artifact_stem(
    asof: str,
    matched_window_with_opentable: bool = False,
    matched_source: str = "ot",
) -> str:
    if matched_window_with_opentable:
        suffix = matched_window_suffix(
            matched_window_with_opentable=matched_window_with_opentable,
            matched_source=matched_source,
        )
        return f"opentable_private_observed_{asof}{suffix}"
    return f"opentable_private_lap_{asof}"


def run_tag_for_mode(
    mode: str,
    use_adapter: bool,
    smooth_cases_window: int,
    matched_window_with_opentable: bool = False,
    matched_source: str = "ot",
) -> str:
    base = f"{mode}{'_adapter' if use_adapter else ''}_w{int(smooth_cases_window)}"
    if matched_window_with_opentable:
        base += matched_window_suffix(
            matched_window_with_opentable=matched_window_with_opentable,
            matched_source=matched_source,
        )
    return base
