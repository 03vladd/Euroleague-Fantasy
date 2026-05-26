## MODIFIED Requirements

### Requirement: DNP risk table derived from feature pipeline, not standalone computation
The DNP risk table (previously specified in `openspec/changes/dnp-aware-sub-advisor/specs/dnp-risk-table/spec.md` as a standalone computation) SHALL instead be derived from the `dnp_rate_last5` and `dnp_rate_last10` columns produced by `build_features.py`.

`optimize_team_v2.py` SHALL print a DNP risk section for the selected team showing each player's `dnp_rate_last5` and `dnp_rate_last10` alongside their adjusted predicted FP, so the manager can see which players were discounted and by how much.

The format SHALL be:
```
DNP RISK (selected team)
Player                    DNP-L5   DNP-L10   Pred FP   Adj FP
------------------------------------------------------------
PLAYER_NAME               0%       10%        18.4      18.4
RISKY_PLAYER              40%      30%        14.0      11.2
```

#### Scenario: DNP risk table printed for selected team
- **WHEN** optimizer produces a valid team selection
- **THEN** a DNP risk table is printed showing all 10 selected players, sorted by dnp_rate_last5 descending

#### Scenario: Zero-risk player shows no adjustment
- **WHEN** a selected player has dnp_rate_last5 == 0.0
- **THEN** their Adj FP equals their Pred FP in the table

#### Scenario: High-risk player shows visible discount
- **WHEN** a selected player has dnp_rate_last5 == 0.4 and DNP_DISCOUNT == 0.5
- **THEN** Adj FP = Pred FP × 0.8 in the table (20% reduction)