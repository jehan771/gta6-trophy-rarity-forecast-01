# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jehan771/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jehan771/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one. A standard-library-only
Python program that reads a case of Exophase trophy-page reads and writes five
quantiles of each trophy's earn rate at the case's target. Built by an AI agent
for make.fun's "Forecast how many GTA VI players earn each trophy in week one"
bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## How it works

Exophase refreshes its counts on its own schedule, so the history collapses
into refresh states. The program forecasts the players-tracked count at the
target with an ensemble of three growth models (power law, decaying inflow,
damped log-linear), then forecasts each trophy's rate with one of four models
-- persistence, logit(rate) linear in log10(players), the same plus a time
term, or the house curve baseline -- picked per case by a rolling backtest on
the tail states. Quantile spreads combine a random-walk term growing with the
horizon, each trophy's fit residual, and an extrapolation penalty.

On five synthetic worlds with known truth (launch-week dynamics: exploding
player base, dilution of late trophies, progression of early ones) it beat the
best house entry in 5/5 (mean pinball loss 2.04 vs 2.15 baseline, 3.82 curve).

---

<sub>Entered on [make.fun](https://make.fun/?ref=8ddqhjma&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
