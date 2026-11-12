# Up-and-Coming Top-10 Forecaster

Entry for the make.fun bounty **roblox-up-and-coming-forecast-01** ("Predict Roblox's
Up-and-Coming top 10"). `node main.ts` reads `/input/history.ndjson` and
`/input/task.json` and writes `/output/forecast.json`: one probability distribution
over games for each of the chart's top 10 places, plus a guessed top 10 in order.
(`INPUT_DIR` / `OUTPUT_DIR` — or the `MAKEFUN_` spellings — override the directories.)

## Approach

The chart churns too fast for persistence: over a 72-hour window most of the top 10
turns over. The model therefore estimates, for every game in the history, a latent
strength on a log-rank scale, `y = -ln(rank)`, and simulates the window hour by hour:

- **State.** Two time-decayed averages of `y` per game: a fast one (half-life 4 h —
  where the game stands now) and a slow one (half-life 160 h — its long-run level).
  Reads where a game is absent count as a rank far past the chart's end, so dropped
  games fade and returning games start from their long-run level.
- **Momentum.** A shrunk, capped slope of `y` over the last 36 h pushes climbers up
  and fallers down, fading with a 35 h time constant.
- **Playing growth.** The chart is sorted by user growth, and it shows: games whose
  concurrent-player count grew over the last ~48 h reach the top 10 far more often
  than decliners at the same rank. A capped log-playing-change term (recent ~14 h vs
  the ~14–62 h before) shifts strength persistently.
- **Simulation.** For each of the window's hours, strengths mean-revert toward the
  long-run level (Ornstein–Uhlenbeck, phi 0.982/h) while Gaussian noise accumulates
  with horizon (sigma_inf 2.2 log-rank units), then the games are ranked. Counting
  places over 1,600 seeded simulations per hour gives each place's probabilities.
  Games whose Roblox `created` date (from the reads' details, when present) is 28+
  days before an hour leave that hour's chart, as on the real chart.
- **Calibration.** A small slice (6%) of each place is blended with the recent
  empirical place-occupancy distribution, probabilities under 0.0006 are dropped,
  and 3% of each place is deliberately left unnamed — the scorer shares it over the
  read's chart, so it is what a brand-new game scores if it breaks into the top 10.

`top10` orders games by expected place under the same distribution. The program is
deterministic (fixed-seed mulberry32 PRNG; no clock, no network, standard library
only) and runs in a few seconds.

## Backtests

Cutting the practice history at 10 points (after reads 10…29 of 34) and scoring the
forecast against the actual later reads with the competition's scoring rule, the
model beats the house persistence baseline at **every** cut: mean log score
**-41.1 vs -49.8** (mean margin **+8.8** per read, summed over the 10 places), with
the smallest margin +0.2 on the shortest (4-read) window and +14.8 on the cut
closest to the practice setup. The house baseline scores -43.51 on the official
practice window; these backtests use only the history the kit ships, so the model's
parameters were chosen on them and nothing else.
