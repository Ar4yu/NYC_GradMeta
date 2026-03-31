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


def normalize_privacy_mode(privacy_mode: str | None) -> str:
    mode = str(privacy_mode or "none").strip().lower()
    if mode in {"", "none", "nonprivate", "non_private"}:
        return "none"
    if mode not in {"event", "restaurant"}:
        raise ValueError(f"Unsupported privacy_mode '{privacy_mode}'. Expected none|event|restaurant.")
    return mode


def normalize_privacy_mechanism(mechanism: str | None) -> str:
    mech = str(mechanism or "gaussian").strip().lower()
    if mech != "gaussian":
        raise ValueError(f"Unsupported privacy mechanism '{mechanism}'. Expected gaussian.")
    return mech


def epsilon_tag(epsilon: float | int | None) -> str:
    if epsilon is None:
        raise ValueError("epsilon is required for DP artifact naming.")
    value = float(epsilon)
    if value <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def privacy_suffix(
    privacy_mode: str | None = "none",
    mechanism: str | None = "gaussian",
    epsilon: float | int | None = None,
) -> str:
    mode = normalize_privacy_mode(privacy_mode)
    if mode == "none":
        return ""
    mech = normalize_privacy_mechanism(mechanism)
    return f"_dp_{mech}_{mode}_eps{epsilon_tag(epsilon)}"


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
    privacy_mode: str | None = "none",
    mechanism: str | None = "gaussian",
    epsilon: float | int | None = None,
) -> str:
    dp_suffix = privacy_suffix(
        privacy_mode=privacy_mode,
        mechanism=mechanism,
        epsilon=epsilon,
    )
    if matched_window_with_opentable:
        suffix = matched_window_suffix(
            matched_window_with_opentable=matched_window_with_opentable,
            matched_source=matched_source,
        )
        return f"opentable_private_observed{dp_suffix}_{asof}{suffix}"
    return f"opentable_private_lap{dp_suffix}_{asof}"


def run_tag_for_mode(
    mode: str,
    use_adapter: bool,
    smooth_cases_window: int,
    matched_window_with_opentable: bool = False,
    matched_source: str = "ot",
    privacy_mode: str | None = "none",
    mechanism: str | None = "gaussian",
    epsilon: float | int | None = None,
) -> str:
    mode_norm = normalize_privacy_mode(privacy_mode)
    base = mode
    if mode_norm == "none" and use_adapter:
        base += "_adapter"
    if mode_norm != "none":
        base += privacy_suffix(
            privacy_mode=mode_norm,
            mechanism=mechanism,
            epsilon=epsilon,
        )
    base += f"_w{int(smooth_cases_window)}"
    if matched_window_with_opentable:
        base += matched_window_suffix(
            matched_window_with_opentable=matched_window_with_opentable,
            matched_source=matched_source,
        )
    return base
