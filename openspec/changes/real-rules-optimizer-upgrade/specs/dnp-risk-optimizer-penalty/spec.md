## ADDED Requirements

### Requirement: DNP risk discount applied to predicted FP before ILP
`optimize_team_v2.py` SHALL apply a multiplicative discount to each player's `predicted_fp` before passing it to the ILP objective:

```
adj_predicted_fp = predicted_fp × max(0, 1 − DNP_DISCOUNT × dnp_rate_last5)
```

`DNP_DISCOUNT` SHALL be a named constant defined at the top of the script, defaulting to `0.5`.

The ILP objective SHALL use `adj_predicted_fp` in place of `predicted_fp`. All display columns (L3, Avg, etc.) SHALL continue to show unadjusted values so the user can audit the raw prediction alongside the adjusted one. The printed table SHALL include an `AdjFP` column showing the discount-adjusted value used by the LP.

If `dnp_rate_last5` is not present in the prediction dataframe (e.g., when running against an older feature file), the discount SHALL be silently skipped and a warning printed.

#### Scenario: High-DNP player is penalised
- **WHEN** a player has `predicted_fp` = 20.0 and `dnp_rate_last5` = 0.6
- **THEN** `adj_predicted_fp` = 20.0 × (1 − 0.5 × 0.6) = 14.0

#### Scenario: Healthy player is unaffected
- **WHEN** a player has `predicted_fp` = 15.0 and `dnp_rate_last5` = 0.0
- **THEN** `adj_predicted_fp` = 15.0

#### Scenario: Extreme DNP rate is clamped
- **WHEN** a player has `dnp_rate_last5` = 1.0 (all DNPs) and `DNP_DISCOUNT` = 0.5
- **THEN** `adj_predicted_fp` = predicted_fp × 0.5 (not negative)

#### Scenario: Missing DNP feature degrades gracefully
- **WHEN** `dnp_rate_last5` column is absent from the prediction CSV
- **THEN** optimizer prints a warning and runs without the discount