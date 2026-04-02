# Thesis Results Preview

Canonical matched experiment:
- Window: 2020-02-29 to 2020-08-05
- Train: 2020-02-29 to 2020-07-08
- Test: 2020-07-09 to 2020-08-05
- Forecast horizon: 28 days
- K: 159

## Main Thesis Table

| condition_label | privacy_mode | epsilon | rmse | mae | mape | sigma_pp | K | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: public-only baseline | none |  | 126.78001661253305 | 113.3240100315639 | 32.739030550752915 |  |  | Public-only matched baseline. |
| B: public + OpenTable (non-private) | none |  | 64.33216967208595 | 49.621803283691406 | 17.279345368032896 |  |  | Non-private OpenTable reference. |
| C: public + OpenTable with Gaussian DP | event | 1.0 | 160.56080313657614 | 143.55575016566686 | 48.146940206628884 | 1.0859030759746926 | 159 | Event-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | event | 2.0 | 147.3097753384828 | 131.25423649379186 | 44.07683699926971 | 0.5429515379873463 | 159 | Event-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | event | 4.0 | 42.72846575897132 | 38.50791713169643 | 11.652532168913075 | 0.27147576899367315 | 159 | Event-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | event | 8.0 | 85.70895575504697 | 70.98228781563895 | 20.14003172780536 | 0.13573788449683657 | 159 | Event-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | event | 16.0 | 81.49119124451356 | 65.13429151262555 | 18.07796712423014 | 0.06786894224841829 | 159 | Event-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | restaurant | 1.0 | 58.65476279205224 | 55.03894805908203 | 17.323558811659215 | 13.69271678577325 | 159 | Restaurant-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | restaurant | 2.0 | 53.68693868398866 | 47.09053911481585 | 13.815561238990087 | 6.846358392886625 | 159 | Restaurant-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | restaurant | 4.0 | 74.7908616696626 | 57.550510951450896 | 20.187274197356196 | 3.4231791964433125 | 159 | Restaurant-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | restaurant | 8.0 | 99.34871593866795 | 83.9949449811663 | 23.757025480564224 | 1.7115895982216562 | 159 | Restaurant-level Gaussian DP. |
| C: public + OpenTable with Gaussian DP | restaurant | 16.0 | 52.789583463394614 | 49.212271554129465 | 15.571686985058216 | 0.8557947991108281 | 159 | Restaurant-level Gaussian DP. |

## Non-Private Matched Runs

| run_name | condition_label | smoothing_window | rmse | mae | mape | notes |
| --- | --- | --- | --- | --- | --- | --- |
| public_only_adapter_w0_matched_ot | A: public-only baseline | 0 | 648.8870052093379 | 628.3805106026786 | 237.51327550615332 | Run metadata JSON missing.; Public-only matched baseline. |
| public_only_adapter_w3_matched_ot | A: public-only baseline | 3 | 4679.261364646586 | 3806.7816494532995 | 1385.1325487793245 | Run metadata JSON missing.; Public-only matched baseline. |
| public_only_adapter_w7_matched_ot | A: public-only baseline | 7 | 126.78001661253305 | 113.3240100315639 | 32.739030550752915 | Public-only matched baseline. |
| public_opentable_adapter_w0_matched_ot | B: public + OpenTable (non-private) | 0 | 191.01385974356987 | 167.18595123291016 | 70.4824134111932 | Run metadata JSON missing.; Non-private OpenTable matched run. |
| public_opentable_adapter_w3_matched_ot | B: public + OpenTable (non-private) | 3 | 2319.0305733072655 | 1780.5190751211983 | 655.8655751489575 | Run metadata JSON missing.; Non-private OpenTable matched run. |
| public_opentable_adapter_w7_matched_ot | B: public + OpenTable (non-private) | 7 | 64.33216967208595 | 49.621803283691406 | 17.279345368032896 | Run metadata JSON missing.; Non-private OpenTable matched run. |

## DP Matched Runs

| run_name | privacy_mode | epsilon | rmse | mae | mape | sigma_pp | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| public_opentable_dp_gaussian_event_eps1_w7_matched_ot | event | 1.0 | 160.56080313657614 | 143.55575016566686 | 48.146940206628884 | 1.0859030759746926 |  |
| public_opentable_dp_gaussian_event_eps2_w7_matched_ot | event | 2.0 | 147.3097753384828 | 131.25423649379186 | 44.07683699926971 | 0.5429515379873463 |  |
| public_opentable_dp_gaussian_event_eps4_w7_matched_ot | event | 4.0 | 42.72846575897132 | 38.50791713169643 | 11.652532168913075 | 0.27147576899367315 |  |
| public_opentable_dp_gaussian_event_eps8_w7_matched_ot | event | 8.0 | 85.70895575504697 | 70.98228781563895 | 20.14003172780536 | 0.13573788449683657 |  |
| public_opentable_dp_gaussian_event_eps16_w7_matched_ot | event | 16.0 | 81.49119124451356 | 65.13429151262555 | 18.07796712423014 | 0.06786894224841829 |  |
| public_opentable_dp_gaussian_restaurant_eps1_w7_matched_ot | restaurant | 1.0 | 58.65476279205224 | 55.03894805908203 | 17.323558811659215 | 13.69271678577325 |  |
| public_opentable_dp_gaussian_restaurant_eps2_w7_matched_ot | restaurant | 2.0 | 53.68693868398866 | 47.09053911481585 | 13.815561238990087 | 6.846358392886625 |  |
| public_opentable_dp_gaussian_restaurant_eps4_w7_matched_ot | restaurant | 4.0 | 74.7908616696626 | 57.550510951450896 | 20.187274197356196 | 3.4231791964433125 |  |
| public_opentable_dp_gaussian_restaurant_eps8_w7_matched_ot | restaurant | 8.0 | 99.34871593866795 | 83.9949449811663 | 23.757025480564224 | 1.7115895982216562 |  |
| public_opentable_dp_gaussian_restaurant_eps16_w7_matched_ot | restaurant | 16.0 | 52.789583463394614 | 49.212271554129465 | 15.571686985058216 | 0.8557947991108281 |  |

## Extraction Notes

- Best non-private RMSE in the matched comparison: `public_opentable_adapter_w7_matched_ot` with RMSE `64.33216967208595`.
- `w=7` is the most thesis-facing operating point because it delivers the strongest matched non-private result and avoids the severe instability seen at `w=3`; `w=0` is usable but weaker than the `w=7` OpenTable run.
- Event-level DP is mixed rather than monotone: the best event-level result is `public_opentable_dp_gaussian_event_eps4_w7_matched_ot` with RMSE `42.72846575897132`, while other event epsilons are weaker than the non-private reference.
- Restaurant-level DP is also mixed: it beats the non-private OpenTable RMSE at eps `1`, `2`, and `16`, but underperforms event-level DP at eps `4` and `8`.
- Directionally, the event-level and restaurant-level DP curves should be described as non-monotonic and implementation-specific rather than as a simple smooth privacy-utility frontier.
