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


def online_artifact_stem(split: str, asof: str, window_days: int, smooth_cases_window: int) -> str:
    return f"{split}_{asof}_history{int(window_days)}_w{int(smooth_cases_window)}"
