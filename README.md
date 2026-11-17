# GTA VI Twitch CCU Forecast — make.fun entry

[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=badge&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mansyur0523/makefun-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=badge&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Mansyur0523/makefun-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=badge&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [gta6-twitch-ccu-forecast-01](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty ($250, 20 paid places): forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), hours 2–24, as five quantiles per hour.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads each case file
in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one forecast per case to
`$MAKEFUN_OUTPUT_DIR` (default `/output`).

The key observation from the 38 launch comparables in the case file: release-day
curves do **not** share one shape when divided by the first hour's print. A game
whose first hour is already huge (global unlock the night before) barely grows,
while a game whose first hour is tiny (rolling midnight, most regions still
locked) can grow a hundredfold — log hour-1 viewers and log peak/hour-1 ratio
correlate at −0.94. So for each forecast hour the program fits, across the
comparables, a log-linear model of that hour's viewers on four log features of
the launch itself — hour-1 viewers, hour-1 live channels, biggest pre-release
hourly audience, and viewers twelve hours before the release day — weighted by
a Gaussian kernel on log hour-1 viewers, so launches of a similar size count
most. The fitted value is the median; the five quantiles are a lognormal band
around it with log spread 0.32. The median is clamped to the comparables'
observed range for that hour (±1 log unit), with a persistence fallback if a
case carries no usable comparables.

## Local validation (practice cases, answers from SullyGnome)

Mean pinball loss on the log scale, scored exactly as the rules define it:

| case | game | loss | 90% coverage | typical miss |
| --- | --- | --- | --- | --- |
| p00 | Marvel's Wolverine | 0.1426 | 23/23 | 24.8% |
| p01 | Red Dead Redemption 2 | 0.1695 | 20/23 | 33.7% |
| p02 | God of War Ragnarök | 0.1153 | 23/23 | 20.5% |
| p03 | Marvel's Spider-Man 2 | 0.1959 | 20/23 | 36.7% |
| **mean** | | **0.1558** | **86/92** | **~29%** |

House baseline on the same practice: loss ~0.387, typical miss ~82.8%,
coverage ~0.783. This entry is ~60% better than baseline on practice loss.

Run it locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic (byte-identical outputs across runs), ~0.07 s per case, outputs
~1.5 KB per case.
