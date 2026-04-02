# Figure Caption Numbers

## Main w=7 non-private comparison

- `public_only_adapter_w7_matched_ot`: RMSE `126.78001661253305`, MAE `113.3240100315639`.
- `public_opentable_adapter_w7_matched_ot`: RMSE `64.33216967208595`, MAE `49.621803283691406`.

## Event-level DP RMSE by epsilon

- eps `1.0`: RMSE `160.56080313657614`.
- eps `2.0`: RMSE `147.3097753384828`.
- eps `4.0`: RMSE `42.72846575897132`.
- eps `8.0`: RMSE `85.70895575504697`.
- eps `16.0`: RMSE `81.49119124451356`.

## Restaurant-level DP RMSE by epsilon

- eps `1.0`: RMSE `58.65476279205224`.
- eps `2.0`: RMSE `53.68693868398866`.
- eps `4.0`: RMSE `74.7908616696626`.
- eps `8.0`: RMSE `99.34871593866795`.
- eps `16.0`: RMSE `52.789583463394614`.

## Factual summaries

- A/B smoothing comparison: In the matched non-private runs, `w=7` is the strongest operating point for the OpenTable condition, while `w=3` is markedly unstable for both public-only and OpenTable runs.
- Event-level DP trend: Event-level RMSE is non-monotonic across epsilon, with the strongest committed event-level result at eps `4` and weaker results at eps `1`, `2`, `8`, and `16`.
- Restaurant-level DP trend: Restaurant-level RMSE is also non-monotonic, with relatively strong committed results at eps `1`, `2`, and `16`, and weaker results at eps `4` and `8`.
