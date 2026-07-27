# risk_profile.py — financial risk-persona battery

## Purpose

Produce a per-model financial **risk persona** — how conservative vs. risk-liberal a model is
when allocating capital — as a parameter vector plus a single 0-100 **conservatism index**.

It exists because `risk_threshold.py` saturated: it scored ~47 on every model. That tool only
probes the **gain domain with clean, symmetric, computable EV**, where there is exactly one
rational answer — every competent model computes the EV sign and acts on it, so the indifference
point lands in the same place for all of them and the score can't discriminate.

The methods here break that crutch the way the behavioral-economics / LLM-elicitation literature
does:

1. a **loss domain** (negative outcomes) — reveals **loss aversion** (λ)
2. **varied / distorted probabilities** — reveals **probability weighting** (α)
3. **switching-point** elicitation — reveals **utility curvature** / risk aversion (σ), where you
   *switch* reveals the attitude, not the EV sign.

Validated discrimination (positive control on Qwen3.6-35B via `--persona`): the gauge spans
**seeking → 40, neutral → 49–52, averse → 85**. The old tool was 47 for all of them.

## Modes (`--mode`, default `all`)

| mode    | instrument | yields |
|---------|-----------|--------|
| `tcn`   | Tanaka-Camerer-Nguyen 3 lottery series (2 gain at different odds + 1 mixed gain/loss) | σ (curvature), λ (loss aversion), α (prob weighting) — **revealed**, feeds the index |
| `hl`    | Holt-Laury 10-row price list (canonical $2.00/$1.60 vs $3.85/$0.10 payoffs) | CRRA `r` — revealed anchor, feeds the index |
| `grable`| Grable-Lytton (1999) 13-item scale (`grable_lytton.json`) | stated risk-tolerance score 13-47 — **stated**, shown for stated-vs-revealed comparison, NOT in the index |
| `framing`| Asian-disease reflection pair + Ellsberg ambiguity pair | reflection effect, ambiguity aversion — consistency probe, reported separately |

## The math (prospect theory, TCN specification)

```
v(x) =  x^(1-σ)            for x >= 0
v(x) = -λ · (-x)^(1-σ)     for x <  0
w(p) =  exp( -(-ln p)^α )                # Prelec; α=1 -> w(p)=p (no distortion)
```
Two-outcome prospects use rank-dependent weighting on the better outcome's probability.

**Instrument generation.** The TCN and Holt-Laury lottery rows are *derived from the equations*,
not transcribed from a paper — transparent and reproducible. Each gain-series row's risky high
payoff is the value that makes an EU agent of a target σ indifferent (σ grid spans −0.25…0.85);
rows sort by ascending payoff so the switch row brackets σ. The mixed series varies the bad-case
loss so the switch brackets λ. Holt-Laury uses its canonical published payoffs. This is legitimate
instrument design (like the old tool's EV ladder); the model's *choices* are the real measurements.

**Fitting.** Each series is one API call; the model returns an A/B choice per numbered row, which
exposes the whole switching pattern (multiple-switching is detected as an inconsistency flag rather
than silently collapsed). Per series we take the median switch point across samples.
- σ, α: jointly solved from the two gain-series indifference points (`fit_sigma_alpha`,
  scipy `least_squares`). **Critical:** the solver is seeded from the EU-implied σ of each series —
  a fixed seed slides into a bad local minimum for risk-seeking (negative-σ) models. The gap
  between the two series' EU-implied σ is the probability-weighting (α) signal.
- λ: closed form from the mixed-series boundary given σ (`lambda_from_series3`).
- CRRA `r`: `crra_from_hl`, using the **normalized** CRRA `u(x)=x^(1-r)/(1-r)` (ln x at r=1).
  Unnormalized `x^(1-r)` flips monotonicity for r>1 and breaks the root bracket — that bug pegged
  early Qwen runs at the −0.9 floor and mislabeled a risk-neutral model "risk-seeking".

**Conservatism index** (`composite_conservatism`): a 0-100 blend of the revealed axes
(σ weight 0.4, λ 0.4, CRRA r 0.2; higher = more conservative). Grable (stated) and framing
(consistency) are reported alongside but excluded from the index.

## Defaults (run bare — `./risk_profile.py`)

These are tuned for a no-arg invocation against the local vLLM server:
- `--mode all`, `--backend openai --url http://localhost:8000`, model auto-detected.
- `--persona neutral` — the real measurement. `averse`/`seeking` are manual positive controls.
- **`--max-tokens` omitted (uncapped)** — the model thinks until EOS and stops naturally, exactly
  like production. A cap is what caused truncation (`null` content) on the thinking model; do not
  reintroduce one. A retry-with-doubled-budget safety net only engages if a cap is set explicitly.
- **`--temperature 0.4`** — a risk *trait* is the model's modal choice; a chat-level 0.7 injected
  switch-row jitter that made the index non-reproducible run-to-run.
- `--samples 4` — switch rows are stable, so 4 medians cleanly; raise for noisier models.
- `--workers 8` — the server's cudagraph capture tops out at batch 8; >8 falls to eager mode
  (~14× slower per the server-tuning notes). Do not raise above 8.

Bare run is ~3 min on Qwen3.6-35B (thinking model, ~36 uncapped calls).

## Reuse

- `risk_threshold.send_claude`, `risk_threshold.parse_json`, `risk_threshold.PERSONAS`
- `test_bias.strip_markdown_fence`, `test_bias.score_label` (via the `t`/`rt` imports)
- `post_openai` is a local variant of `send_openai` that detects truncated/`null` content
  (so it can retry) instead of crashing in `strip_markdown_fence`.

## grable_lytton.json

Verbatim Grable & Lytton (1999) 13-item scale as reproduced by Rutgers/NJAES (source URL in the
file). Items, options, and per-option point weights; score range 13-47 with 5 category bands.
If the file is absent, the `grable` mode reports n/a and the rest of the battery still runs.

## Self-test

`./risk_profile.py --selftest` runs the estimator against synthetic known-parameter PT agents
(no API calls) and checks σ/λ recovery. Use it after any change to the math. σ recovers exactly;
λ recovers within the loss-grid discretization (~0.3, same nature as Holt-Laury's CRRA intervals).

## Usage

```bash
./risk_profile.py                         # full profile, auto-detect model on :8000 (~3 min)
./risk_profile.py --mode tcn --verbose    # just prospect theory, watch each call
./risk_profile.py --score-only            # print only the 0-100 conservatism index
./risk_profile.py --persona averse        # positive control: index should rise
./risk_profile.py --backend claude --model opus   # score Opus via claude -p
./risk_profile.py --selftest              # validate estimator math, no model calls
./risk_profile.py -o qwen_profile.json    # also write the full vector + switch rows
```

## Gotchas / history

- Do not run two batteries against `:8000` concurrently (server contention).
- A near-neutral model (Qwen) lands the index around 50 and the value jitters a couple points
  run-to-run because it sits in the indifference band; models with a real disposition (high λ, or
  the averse/seeking personas) separate cleanly. That is the instrument working, not saturating.
- Series 2 (high-probability gain series) has a deliberately narrow payoff range — that low
  sensitivity is what identifies α, but it also makes its per-row σ steps large, so its switch row
  is the noisiest input. Most of the index's stability comes from σ-series-1 + λ + CRRA.
