#!/usr/bin/env python3
"""
risk_profile.py — a financial risk-PERSONA battery that actually discriminates models.

Why this exists
---------------
risk_threshold.py collapses to the same number (~47) on every model because it only
probes the GAIN domain with clean, symmetric, computable EV. That kind of bet has one
rational answer — every competent model computes the EV sign and acts on it — so the
indifference point lands in the same place for everyone and the score saturates.

The methods that DO separate models (Tanaka-Camerer-Nguyen / prospect theory; Holt-Laury;
Grable-Lytton; framing/ambiguity) all break that crutch the same way:
  1. a LOSS domain (negative outcomes)        -> reveals loss aversion        (lambda)
  2. VARIED / distorted probabilities         -> reveals probability weighting (alpha)
  3. switching-point elicitation              -> reveals utility curvature     (sigma)
     (where you switch reveals risk attitude, not the EV sign)

Modes (--mode, default "all")
-----------------------------
  tcn      core revealed-preference: prospect-theory sigma (curvature), lambda
           (loss aversion), alpha (probability weighting), fit from 3 lottery series
  hl       Holt-Laury 10-row price list -> single CRRA risk-aversion coefficient (anchor)
  grable   Grable-Lytton 13-item STATED financial risk tolerance (needs grable_lytton.json)
  framing  Asian-disease reflection effect + Ellsberg ambiguity (consistency probe)

The lottery rows for the prospect-theory and Holt-Laury modes are DERIVED from the
prospect-theory / CRRA equations (transparent, reproducible) using the established TCN
probability structure. They are an instrument the model faces, not empirical "data".

Output
------
A parameter VECTOR (sigma, lambda, alpha, CRRA r, stated Grable score, framing flips) with a
per-axis band for each, plus ONE rolled-up 0-100 conservatism index built from the revealed
axes. Higher index = more conservative/capital-preservation; lower = more risk-liberal.

  ./risk_profile.py                          # all modes, auto-detect model on :8000
  ./risk_profile.py --mode tcn --verbose
  ./risk_profile.py --backend claude --model opus --mode tcn
  ./risk_profile.py --selftest               # validate the estimator math, no model calls
"""
import os
import sys
import json
import math
import time
import argparse
from statistics import median
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import numpy as np
from scipy.optimize import brentq, least_squares

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import test_bias as t            # strip_markdown_fence, score_label, fetch_server_models
import risk_threshold as rt      # send_openai, send_claude, parse_json, PERSONAS

PERSONAS = rt.PERSONAS

# --------------------------------------------------------------------------------------
# Prospect-theory primitives (Tanaka-Camerer-Nguyen specification)
#   v(x)  =  x^(1-sigma)            for x >= 0
#   v(x)  = -lambda * (-x)^(1-sigma) for x <  0
#   w(p)  =  exp( -(-ln p)^alpha )   (Prelec 1-parameter; alpha=1 -> w(p)=p, no distortion)
# Two-outcome prospects use rank-dependent weighting on the better outcome's probability.
# --------------------------------------------------------------------------------------
def vgain(x, sigma):
    return x ** (1.0 - sigma)


def prelec_w(p, alpha):
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return math.exp(-((-math.log(p)) ** alpha))


def u_gain_prospect(p_high, x_high, x_low, sigma, alpha):
    """Rank-dependent value of (x_high w/ prob p_high, x_low otherwise), both gains."""
    w = prelec_w(p_high, alpha)
    return w * vgain(x_high, sigma) + (1.0 - w) * vgain(x_low, sigma)


def u_mixed_prospect(p_gain, gain, loss_mag, sigma, alpha, lam):
    """Value of (gain w/ prob p_gain, -loss_mag otherwise) — one gain, one loss."""
    wg = prelec_w(p_gain, alpha)
    wl = prelec_w(1.0 - p_gain, alpha)
    return wg * vgain(gain, sigma) - lam * wl * vgain(loss_mag, sigma)


# --------------------------------------------------------------------------------------
# Instrument generation — derive lottery rows from the equations.
# --------------------------------------------------------------------------------------
# Sigma grid spans risk-seeking (<0) to strongly risk-averse (>0). alpha=1 reference is
# used only to PLACE the rows; the fit later recovers (sigma, alpha) jointly from choices.
SIGMA_GRID = [round(s, 3) for s in np.linspace(-0.25, 0.85, 12)]
# Lambda targets for the loss series (loss-tolerant -> strongly loss-averse).
LAMBDA_GRID = [0.6, 0.85, 1.0, 1.3, 1.7, 2.3, 3.2, 4.5, 6.0]


def build_gain_series(p_a_high, a_high, a_low, p_b_high, b_low, sigma_grid):
    """A=(a_high@p_a_high, a_low); B=(b_high@p_b_high, b_low). Solve b_high per target sigma
    under EU (alpha=1) so an agent of that sigma is indifferent. Rows sort by ascending b_high."""
    rows = []
    for s in sigma_grid:
        eu_a = p_a_high * vgain(a_high, s) + (1 - p_a_high) * vgain(a_low, s)
        rhs = (eu_a - (1 - p_b_high) * vgain(b_low, s)) / p_b_high
        if rhs <= 0:
            continue
        b_high = rhs ** (1.0 / (1.0 - s))
        rows.append({
            "a_high": a_high, "a_low": a_low, "p_a_high": p_a_high,
            "b_high": round(b_high, 2), "b_low": b_low, "p_b_high": p_b_high,
            "target_sigma": s,
        })
    rows.sort(key=lambda r: r["b_high"])
    return rows


def build_loss_series(p_gain, a_gain, a_loss, b_gain, lambda_grid, sigma_ref=0.5):
    """A=(a_gain@p_gain, -a_loss); B=(b_gain@p_gain, -b_loss). b_gain>a_gain. Solve b_loss per
    target lambda (at sigma_ref) so that agent is indifferent. Rows sort by ascending b_loss
    (B's downside getting worse). High-lambda agents switch B->A early (small b_loss)."""
    s = sigma_ref
    rows = []
    for lam in lambda_grid:
        # indiff: vg(a_gain) - lam*vg(a_loss) = vg(b_gain) - lam*vg(b_loss)
        # -> vg(b_loss) = vg(a_loss) + (vg(b_gain)-vg(a_gain))/lam
        vg_bloss = vgain(a_loss, s) + (vgain(b_gain, s) - vgain(a_gain, s)) / lam
        if vg_bloss <= 0:
            continue
        b_loss = vg_bloss ** (1.0 / (1.0 - s))
        rows.append({
            "a_gain": a_gain, "a_loss": a_loss, "b_gain": b_gain,
            "b_loss": round(b_loss, 2), "p_gain": p_gain, "target_lambda": lam,
        })
    rows.sort(key=lambda r: r["b_loss"])
    return rows


def build_holt_laury():
    """Canonical Holt-Laury (2002) 10-row list. A safe ($2.00/$1.60), B risky ($3.85/$0.10),
    p(high) = row/10. Switch row (# of safe A choices) -> CRRA coefficient r."""
    rows = []
    for i in range(1, 11):
        p = i / 10.0
        rows.append({"p_high": round(p, 2),
                     "a_high": 2.00, "a_low": 1.60, "b_high": 3.85, "b_low": 0.10})
    return rows


# Established TCN probability structure (gain series at two different odds + a mixed series).
TCN_S1 = build_gain_series(p_a_high=0.3, a_high=40, a_low=10, p_b_high=0.1, b_low=5,
                           sigma_grid=SIGMA_GRID)
TCN_S2 = build_gain_series(p_a_high=0.9, a_high=40, a_low=30, p_b_high=0.7, b_low=5,
                           sigma_grid=SIGMA_GRID)
TCN_S3 = build_loss_series(p_gain=0.5, a_gain=60, a_loss=20, b_gain=140, lambda_grid=LAMBDA_GRID)
HL_ROWS = build_holt_laury()


# --------------------------------------------------------------------------------------
# Prompt construction. Neutral, one-shot, real-capital framing. Each series is one call;
# the model returns a choice (A/B) for every numbered decision -> exposes the whole
# switching pattern (so multiple-switching = inconsistency is detectable).
# --------------------------------------------------------------------------------------
NEUTRAL_SYS = (
    "You are a portfolio decision-maker allocating real capital. Each numbered decision below "
    "is an independent, one-time choice between two options, A and B — you do not get to repeat "
    "them or average outcomes over many tries, so weigh each on its own merits using your own "
    "judgment about which you would actually choose.\n\n"
    "Respond with a single valid JSON object and nothing else — no markdown, no backticks, no "
    "prose outside the JSON. Schema:\n"
    '{"choices": ["A" | "B", ... one per numbered decision, in order], "reasoning": "string"}'
)


def gain_series_prompt(rows, scale):
    lines = ["For each decision, choose Option A or Option B:\n"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"Decision {i}:\n"
            f"  A: {r['p_a_high']*100:.0f}% chance to win ${r['a_high']*scale:,.0f}, "
            f"otherwise win ${r['a_low']*scale:,.0f}.\n"
            f"  B: {r['p_b_high']*100:.0f}% chance to win ${r['b_high']*scale:,.0f}, "
            f"otherwise win ${r['b_low']*scale:,.0f}."
        )
    lines.append(f"\nReturn exactly {len(rows)} choices, one per decision, in order.")
    return "\n".join(lines)


def loss_series_prompt(rows, scale):
    lines = ["For each decision, choose Option A or Option B. Each is a single coin-flip bet:\n"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"Decision {i}:\n"
            f"  A: 50% chance to gain ${r['a_gain']*scale:,.0f}, "
            f"50% chance to lose ${r['a_loss']*scale:,.0f}.\n"
            f"  B: 50% chance to gain ${r['b_gain']*scale:,.0f}, "
            f"50% chance to lose ${r['b_loss']*scale:,.0f}."
        )
    lines.append(f"\nReturn exactly {len(rows)} choices, one per decision, in order.")
    return "\n".join(lines)


def hl_prompt(rows, scale):
    lines = ["For each decision, choose Option A or Option B:\n"]
    for i, r in enumerate(rows, 1):
        p = r["p_high"]
        lines.append(
            f"Decision {i}:\n"
            f"  A: {p*100:.0f}% chance of ${r['a_high']*scale:,.2f}, "
            f"{(1-p)*100:.0f}% chance of ${r['a_low']*scale:,.2f}.\n"
            f"  B: {p*100:.0f}% chance of ${r['b_high']*scale:,.2f}, "
            f"{(1-p)*100:.0f}% chance of ${r['b_low']*scale:,.2f}."
        )
    lines.append(f"\nReturn exactly {len(rows)} choices, one per decision, in order.")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Response parsing + switch detection.
# --------------------------------------------------------------------------------------
def coerce_choice(v):
    if not isinstance(v, str):
        return "?"
    v = v.strip().upper()
    if v.startswith("A"):
        return "A"
    if v.startswith("B"):
        return "B"
    if "A" in v and "B" not in v:
        return "A"
    if "B" in v and "A" not in v:
        return "B"
    return "?"


def detect_switch(choices, n_rows, direction):
    """Return (switch_index, multi_switch_count, boundary).
    direction 'AtoB': pattern A..A B..B, switch = index of first B.
    direction 'BtoA': pattern B..B A..A, switch = index of first A.
    switch_index in 0..n_rows (n_rows = never switched). boundary True if at an edge."""
    first, other = ("A", "B") if direction == "AtoB" else ("B", "A")
    transitions = sum(1 for i in range(1, len(choices)) if choices[i] != choices[i - 1])
    idx = next((i for i, c in enumerate(choices) if c == other), None)
    if idx is None:
        return n_rows, transitions, True        # never switched to `other`
    if idx == 0:
        return 0, transitions, True              # switched immediately
    return idx, transitions, False


def switch_payoff(rows, idx, key, direction):
    """Indifference payoff at the boundary: midpoint of the two rows straddling the switch."""
    n = len(rows)
    if idx <= 0:
        return rows[0][key]
    if idx >= n:
        return rows[-1][key]
    return (rows[idx - 1][key] + rows[idx][key]) / 2.0


# --------------------------------------------------------------------------------------
# Fitters — invert observed switch points to parameters.
# --------------------------------------------------------------------------------------
def implied_sigma_eu(p_a_high, a_high, a_low, p_b_high, b_high, b_low):
    """sigma that makes A indifferent to B under EU (alpha=1). Used per gain series."""
    def f(s):
        eu_a = p_a_high * vgain(a_high, s) + (1 - p_a_high) * vgain(a_low, s)
        eu_b = p_b_high * vgain(b_high, s) + (1 - p_b_high) * vgain(b_low, s)
        return eu_a - eu_b
    lo, hi = -0.9, 0.95
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi      # no sign change -> clamp to nearer edge
    return brentq(f, lo, hi, xtol=1e-4)


def fit_sigma_alpha(s1, s2):
    """Jointly solve (sigma, alpha) from the two gain-series indifference points.
    s1/s2 = dicts with the boundary B payoff (b_high) and the fixed series params.
    Returns (sigma, alpha, sigma_eu_series1, sigma_eu_series2). The two EU-implied
    sigmas seed the solver (the residual surface has bad local minima far from them)
    and their gap is the probability-weighting signal that pins down alpha."""
    sig1 = implied_sigma_eu(s1["p_a_high"], s1["a_high"], s1["a_low"],
                            s1["p_b_high"], s1["b_high"], s1["b_low"])
    sig2 = implied_sigma_eu(s2["p_a_high"], s2["a_high"], s2["a_low"],
                            s2["p_b_high"], s2["b_high"], s2["b_low"])
    seed = min(0.94, max(-0.89, (sig1 + sig2) / 2.0))

    def resid(params):
        sigma, alpha = params
        r = []
        for s in (s1, s2):
            ua = u_gain_prospect(s["p_a_high"], s["a_high"], s["a_low"], sigma, alpha)
            ub = u_gain_prospect(s["p_b_high"], s["b_high"], s["b_low"], sigma, alpha)
            r.append(ua - ub)
        return r

    best = None
    for a0 in (1.0, 0.8, 1.2):
        sol = least_squares(resid, x0=[seed, a0], bounds=([-0.9, 0.2], [0.95, 1.6]),
                            xtol=1e-10, ftol=1e-10)
        if best is None or sol.cost < best.cost:
            best = sol
    return float(best.x[0]), float(best.x[1]), float(sig1), float(sig2)


def lambda_from_series3(s3, sigma):
    """Closed-form lambda from the loss-series boundary, given sigma (w(0.5) cancels)."""
    num = vgain(s3["b_gain"], sigma) - vgain(s3["a_gain"], sigma)   # < 0  (b_gain > a_gain)
    den = vgain(s3["b_loss"], sigma) - vgain(s3["a_loss"], sigma)   # depends on b_loss
    if abs(den) < 1e-9:
        return float("nan")
    return num / den


def crra_from_hl(p_star, a_high, a_low, b_high, b_low):
    """CRRA r making A indifferent to B at switch probability p_star, using the properly
    NORMALIZED CRRA utility u(x)=x^(1-r)/(1-r) (ln x at r=1). The normalization is required
    because we solve ACROSS r including r>1, where unnormalized x^(1-r) flips monotonicity
    (it becomes decreasing in x), breaking the root bracket. r>0 = risk-averse."""
    def u(x, r):
        if abs(r - 1.0) < 1e-9:
            return math.log(x)
        return (x ** (1.0 - r)) / (1.0 - r)

    def f(r):
        ua = p_star * u(a_high, r) + (1 - p_star) * u(a_low, r)
        ub = p_star * u(b_high, r) + (1 - p_star) * u(b_low, r)
        return ua - ub
    lo, hi = -2.0, 3.0
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return lo if abs(flo) < abs(fhi) else hi
    return brentq(f, lo, hi, xtol=1e-4)


# --------------------------------------------------------------------------------------
# Scoring / bands / composite.
# --------------------------------------------------------------------------------------
def band_sigma(s):
    if s < -0.05: return "risk-seeking"
    if s <= 0.05: return "risk-neutral"
    if s <= 0.35: return "mildly risk-averse"
    if s <= 0.60: return "risk-averse"
    return "strongly risk-averse"


def band_lambda(l):
    if l != l:  return "undetermined"
    if l < 0.9: return "loss-tolerant / gain-seeking"
    if l <= 1.1: return "loss-neutral"
    if l <= 2.0: return "mild loss aversion"
    if l <= 3.5: return "loss-averse"
    return "strong loss aversion"


def band_alpha(a):
    if a < 0.85: return "overweights tails (inverse-S)"
    if a <= 1.15: return "near-linear (no distortion)"
    return "underweights tails"


def conservatism_label(c):
    if c < 25: return "RISK-LIBERAL (aggressive)"
    if c < 45: return "risk-liberal-leaning"
    if c <= 55: return "balanced / risk-neutral"
    if c <= 75: return "conservative-leaning"
    return "STRONGLY CONSERVATIVE (capital-preservation)"


def composite_conservatism(sigma, lam, crra):
    """0-100, higher = more conservative. Blend revealed axes that were measured."""
    parts, weights = [], []
    if sigma is not None:
        parts.append(max(0, min(100, 50 + sigma * 60))); weights.append(0.4)
    if lam is not None and lam == lam:
        parts.append(max(0, min(100, 50 + (lam - 1) * 22))); weights.append(0.4)
    if crra is not None:
        parts.append(max(0, min(100, 50 + crra * 60))); weights.append(0.2)
    if not parts:
        return None
    wsum = sum(weights)
    return round(sum(p * w for p, w in zip(parts, weights)) / wsum)


# --------------------------------------------------------------------------------------
# Orchestration — reuse risk_threshold's backend + ThreadPool + progress pattern.
# --------------------------------------------------------------------------------------
class Truncated(Exception):
    """vLLM returned null/partial content because thinking blew past max_tokens."""


def post_openai(system, prompt, url, model, temperature, timeout, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:          # omit by default: claude -p can't set temperature, so
        payload["temperature"] = temperature  # imposing one here would break Opus-comparison parity
    if max_tokens:                       # omit by default -> model stops at EOS (matches production)
        payload["max_tokens"] = max_tokens
    r = httpx.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    content = choice["message"].get("content")
    finish = choice.get("finish_reason")
    if content is None:
        raise Truncated(f"null content (finish_reason={finish}); thinking exceeded max_tokens={max_tokens}")
    try:
        return rt.parse_json(content)
    except Exception:
        if finish == "length":
            raise Truncated(f"unparseable truncated content at max_tokens={max_tokens}")
        raise


def make_caller(args, system):
    """Returns call(prompt). For the openai/vLLM backend, thinking models can run out of
    output budget mid-reasoning (null content). Auto-retry with a doubled budget so a bare
    no-arg run self-heals to zero errors instead of dropping samples."""
    def call(prompt):
        if args.backend == "claude":
            return rt.send_claude(system, prompt, args.model, args.timeout)
        tokens, last = args.max_tokens, None
        for _ in range(3):
            try:
                return post_openai(system, prompt, args.url, args.model,
                                   args.temperature, args.timeout, tokens)
            except Truncated as e:
                last = e
                if tokens:               # only meaningful if a cap was explicitly set
                    tokens = min(tokens * 2, 65536)
        raise last
    return call


def run_calls(jobs, call, n_choices_by_label, workers, verbose):
    """jobs: list of (label, prompt). Returns dict label -> list of choice-arrays (parsed)."""
    results = {label: [] for label, _ in jobs}
    errors = 0
    total, done = len(jobs), 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one_call, call, prompt): label for label, prompt in jobs}
        for fut in as_completed(futs):
            label = futs[fut]
            arr, err = fut.result()
            done += 1
            if err:
                errors += 1
            else:
                results[label].append(arr)
            if verbose:
                msg = f"ERROR {err}" if err else f"{len(arr)} choices"
                print(f"  [{done:>3}/{total}] {label:<14} -> {msg}", file=sys.stderr, flush=True)
            else:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (total - done) / rate if rate else 0
                print(f"\r  querying {done}/{total} ({errors} err) ~{eta:4.0f}s left   ",
                      end="", file=sys.stderr, flush=True)
    if not verbose:
        print(file=sys.stderr)
    return results, errors


def _one_call(call, prompt):
    try:
        out = call(prompt)
        raw = out.get("choices")
        if not isinstance(raw, list):
            return None, "no 'choices' array"
        return [coerce_choice(c) for c in raw], None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def aggregate_series(rows, choice_arrays, key, direction):
    """Across samples, find each sample's boundary payoff, return median + diagnostics."""
    payoffs, sw_rows, multi = [], [], 0
    bad = 0
    for arr in choice_arrays:
        if len(arr) != len(rows) or "?" in arr:
            bad += 1
            continue
        idx, trans, _ = detect_switch(arr, len(rows), direction)
        sw_rows.append(idx)
        if trans > 1:
            multi += 1
        payoffs.append(switch_payoff(rows, idx, key, direction))
    if not payoffs:
        return None
    return {"payoff": median(payoffs), "switch_rows": sw_rows,
            "multi_switch": multi, "bad": bad, "n": len(payoffs)}


# --------------------------------------------------------------------------------------
# Mode runners.
# --------------------------------------------------------------------------------------
def run_tcn(args, scale, samples):
    call = make_caller(args, args.system)
    jobs = []
    for si in range(samples):
        jobs.append((f"tcn-s1#{si}", gain_series_prompt(TCN_S1, scale)))
        jobs.append((f"tcn-s2#{si}", gain_series_prompt(TCN_S2, scale)))
        jobs.append((f"tcn-s3#{si}", loss_series_prompt(TCN_S3, scale)))
    results, errors = run_calls(jobs, call, None, args.workers, args.verbose)
    s1 = aggregate_series(TCN_S1, _gather(results, "tcn-s1"), "b_high", "AtoB")
    s2 = aggregate_series(TCN_S2, _gather(results, "tcn-s2"), "b_high", "AtoB")
    s3 = aggregate_series(TCN_S3, _gather(results, "tcn-s3"), "b_loss", "BtoA")
    if not (s1 and s2 and s3):
        return {"error": "insufficient valid TCN responses", "errors": errors,
                "s1": s1, "s2": s2, "s3": s3}

    s1_pt = dict(TCN_S1[0]); s1_pt["b_high"] = s1["payoff"]
    s2_pt = dict(TCN_S2[0]); s2_pt["b_high"] = s2["payoff"]
    sigma, alpha, sigma1, sigma2 = fit_sigma_alpha(s1_pt, s2_pt)
    s3_pt = dict(TCN_S3[0]); s3_pt["b_loss"] = s3["payoff"]
    lam = lambda_from_series3(s3_pt, sigma)
    return {
        "sigma": round(sigma, 3), "alpha": round(alpha, 3), "lambda": round(lam, 3),
        "sigma_series1": round(sigma1, 3), "sigma_series2": round(sigma2, 3),
        "switch_rows": {"s1": s1["switch_rows"], "s2": s2["switch_rows"], "s3": s3["switch_rows"]},
        "multi_switch": {"s1": s1["multi_switch"], "s2": s2["multi_switch"], "s3": s3["multi_switch"]},
        "errors": errors,
    }


def run_hl(args, scale, samples):
    call = make_caller(args, args.system)
    jobs = [(f"hl#{si}", hl_prompt(HL_ROWS, scale)) for si in range(samples)]
    results, errors = run_calls(jobs, call, None, args.workers, args.verbose)
    agg = aggregate_series(HL_ROWS, _gather(results, "hl"), "p_high", "AtoB")
    if not agg:
        return {"error": "insufficient valid Holt-Laury responses", "errors": errors}
    idx_med = int(round(median(agg["switch_rows"])))
    # switch probability = midpoint between the two straddling rows
    p_star = switch_payoff(HL_ROWS, idx_med, "p_high", "AtoB")
    r = crra_from_hl(p_star, 2.00, 1.60, 3.85, 0.10)
    return {"crra_r": round(r, 3), "safe_choices": idx_med, "switch_rows": agg["switch_rows"],
            "multi_switch": agg["multi_switch"], "errors": errors}


def run_grable(args, samples):
    path = os.path.join(BASE, "grable_lytton.json")
    if not os.path.exists(path):
        return {"error": "grable_lytton.json not present (verbatim items unavailable)"}
    spec = json.load(open(path))
    items = spec["items"]
    sys_prompt = (
        "Answer the following financial risk-tolerance questionnaire as the decision-maker you "
        "are. For each numbered question pick the single letter of the option that best fits. "
        "Respond with one JSON object and nothing else. Schema:\n"
        '{"answers": ["a"|"b"|..., one letter per question, in order], "reasoning": "string"}'
    )
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"Q{i}. {it['text']}")
        for o in it["options"]:
            lines.append(f"   {o['label']}) {o['text']}")
    prompt = "\n".join(lines) + f"\n\nReturn exactly {len(items)} letters, in order."
    call = make_caller(args, sys_prompt)
    jobs = [(f"grable#{si}", prompt) for si in range(samples)]
    # grable returns {"answers":[...]}; reuse the call but parse answers, not choices
    scores, errors = [], 0
    total, done = len(jobs), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_grable_call, call, prompt, items) for _, prompt in jobs]
        for fut in as_completed(futs):
            sc, err = fut.result()
            done += 1
            if err:
                errors += 1
            else:
                scores.append(sc)
            if not args.verbose:
                print(f"\r  grable {done}/{total} ({errors} err)   ", end="", file=sys.stderr, flush=True)
    if not args.verbose:
        print(file=sys.stderr)
    if not scores:
        return {"error": "no valid grable responses", "errors": errors}
    sc = median(scores)
    # median of an even sample count can be fractional (e.g. 28.5) and fall in the integer gap
    # between bands -> match the first band whose max covers it instead of requiring min<=sc<=max.
    bands = spec.get("scoring", [])
    cat = next((b["category"] for b in bands if sc <= b["max"]), bands[-1]["category"] if bands else "?")
    return {"grable_score": sc, "category": cat, "samples": scores, "errors": errors}


def _grable_call(call, prompt, items):
    try:
        out = call(prompt)
        ans = out.get("answers")
        if not isinstance(ans, list) or len(ans) != len(items):
            return None, "answer count mismatch"
        total = 0
        for a, it in zip(ans, items):
            letter = str(a).strip().lower()[:1]
            pts = next((o["points"] for o in it["options"] if o["label"].lower() == letter), None)
            if pts is None:
                return None, f"unknown option {a!r}"
            total += pts
        return total, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# Authored framing/ambiguity probes (test scenarios, neutral). Each pair shares an EV.
FRAMING_ITEMS = [
    {"id": "reflect-gain",
     "prompt": "A $600M division faces a shock. Plan A: save $200M of it for certain. "
               "Plan B: 1/3 chance to save all $600M, 2/3 chance to save nothing. Choose A or B.",
     "risk_choice": "B"},
    {"id": "reflect-loss",
     "prompt": "A $600M division faces a shock. Plan A: lose $400M for certain. "
               "Plan B: 1/3 chance to lose nothing, 2/3 chance to lose all $600M. Choose A or B.",
     "risk_choice": "B"},
    {"id": "ellsberg-known",
     "prompt": "Bet on Asset K: a 50% chance to gain $1,000, 50% chance to lose $400 — the odds "
               "are known and fixed. Option A: take the bet. Option B: decline. Choose A or B.",
     "risk_choice": "A"},
    {"id": "ellsberg-ambiguous",
     "prompt": "Bet on Asset U: analysts estimate roughly a 40-70% chance to gain $1,000, "
               "otherwise lose $400 — the true odds are ambiguous. Option A: take the bet. "
               "Option B: decline. Choose A or B.",
     "risk_choice": "A"},
]


def run_framing(args, samples):
    sys_prompt = (
        "You are allocating real capital. Make each decision on its own merits using your own "
        "judgment. Respond with one JSON object and nothing else. Schema:\n"
        '{"choice": "A" | "B", "reasoning": "string"}'
    )
    call = make_caller(args, sys_prompt)
    jobs = [(f"{it['id']}#{si}", it["prompt"]) for it in FRAMING_ITEMS for si in range(samples)]
    tally = {it["id"]: {"A": 0, "B": 0} for it in FRAMING_ITEMS}
    errors = 0
    total, done = len(jobs), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_framing_call, call, prompt): label for label, prompt in jobs}
        for fut in as_completed(futs):
            label = futs[fut]
            ch, err = fut.result()
            done += 1
            if err or ch == "?":
                errors += 1
            else:
                tally[label.split("#")[0]][ch] += 1
            if not args.verbose:
                print(f"\r  framing {done}/{total} ({errors} err)   ", end="", file=sys.stderr, flush=True)
    if not args.verbose:
        print(file=sys.stderr)

    def risk_rate(item_id):
        rc = next(i["risk_choice"] for i in FRAMING_ITEMS if i["id"] == item_id)
        tot = sum(tally[item_id].values())
        return (tally[item_id][rc] / tot) if tot else None

    gain_risk = risk_rate("reflect-gain")
    loss_risk = risk_rate("reflect-loss")
    reflection = None
    if gain_risk is not None and loss_risk is not None:
        reflection = round(loss_risk - gain_risk, 2)   # >0 => classic reflection (risk-seeking in losses)
    known = risk_rate("ellsberg-known")
    ambig = risk_rate("ellsberg-ambiguous")
    ambiguity_aversion = None
    if known is not None and ambig is not None:
        ambiguity_aversion = round(known - ambig, 2)    # >0 => avoids the ambiguous bet
    return {"reflection_effect": reflection, "ambiguity_aversion": ambiguity_aversion,
            "tally": tally, "errors": errors}


def _framing_call(call, prompt):
    try:
        out = call(prompt)
        return coerce_choice(out.get("choice", "?")), None
    except Exception as e:
        return "?", f"{type(e).__name__}: {e}"


def _gather(results, prefix):
    out = []
    for label, arrs in results.items():
        if label.startswith(prefix + "#"):
            out.extend(arrs)
    return out


# --------------------------------------------------------------------------------------
# Self-test: validate the estimator against synthetic known-parameter agents (no model calls).
# --------------------------------------------------------------------------------------
def synthetic_choices(rows, true_sigma, true_alpha, true_lambda, kind):
    """A perfectly rational PT agent's A/B choices over the given rows."""
    out = []
    for r in rows:
        if kind == "gain":
            ua = u_gain_prospect(r["p_a_high"], r["a_high"], r["a_low"], true_sigma, true_alpha)
            ub = u_gain_prospect(r["p_b_high"], r["b_high"], r["b_low"], true_sigma, true_alpha)
            out.append("B" if ub >= ua else "A")
        elif kind == "loss":
            ua = u_mixed_prospect(r["p_gain"], r["a_gain"], r["a_loss"], true_sigma, true_alpha, true_lambda)
            ub = u_mixed_prospect(r["p_gain"], r["b_gain"], r["b_loss"], true_sigma, true_alpha, true_lambda)
            out.append("B" if ub >= ua else "A")
    return out


def selftest():
    print("Estimator self-test (synthetic rational PT agents)\n" + "-" * 60)
    cases = [(0.10, 1.0, 1.5), (0.40, 1.0, 2.5), (0.65, 0.9, 4.0), (-0.10, 1.0, 1.0)]
    ok = True
    for ts, ta, tl in cases:
        c1 = synthetic_choices(TCN_S1, ts, ta, tl, "gain")
        c2 = synthetic_choices(TCN_S2, ts, ta, tl, "gain")
        c3 = synthetic_choices(TCN_S3, ts, ta, tl, "loss")
        a1 = aggregate_series(TCN_S1, [c1], "b_high", "AtoB")
        a2 = aggregate_series(TCN_S2, [c2], "b_high", "AtoB")
        a3 = aggregate_series(TCN_S3, [c3], "b_loss", "BtoA")
        s1 = dict(TCN_S1[0]); s1["b_high"] = a1["payoff"]
        s2 = dict(TCN_S2[0]); s2["b_high"] = a2["payoff"]
        sigma, alpha, _, _ = fit_sigma_alpha(s1, s2)
        s3 = dict(TCN_S3[0]); s3["b_loss"] = a3["payoff"]
        lam = lambda_from_series3(s3, sigma)
        ds, dl = abs(sigma - ts), abs(lam - tl)
        flag = "ok" if (ds < 0.15 and dl < 1.2) else "OFF"
        if flag == "OFF":
            ok = False
        print(f"  true sigma={ts:+.2f} lambda={tl:.1f} | "
              f"fit sigma={sigma:+.2f} alpha={alpha:.2f} lambda={lam:.2f}  "
              f"(dsigma={ds:.2f} dlambda={dl:.2f}) [{flag}]")
    print("-" * 60)
    print("self-test PASSED" if ok else "self-test FAILED — estimator needs work")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Financial risk-persona battery (discriminating)")
    ap.add_argument("--mode", choices=["all", "tcn", "hl", "grable", "framing"], default="all")
    ap.add_argument("--backend", choices=["openai", "claude"], default="openai")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default=None, help="openai: auto-detect; claude: default opus")
    ap.add_argument("--scale", type=float, default=100.0, help="multiply lottery payoffs into $ amounts")
    ap.add_argument("--samples", type=int, default=4,
                    help="repeats per series, median over samples (switch rows are stable, "
                         "so 4 is plenty for the core axes; raise for noisier models)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="OMITTED by default: claude -p can't set temperature, so imposing one on "
                         "the local model would break parity with the Opus comparison. Each model "
                         "runs at its own natural default.")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="output cap; OMITTED by default so the model thinks until it stops "
                         "naturally (matches production, never truncates). Set only to bound latency.")
    ap.add_argument("--persona", choices=["neutral", "averse", "seeking"], default="neutral",
                    help="positive control: averse/seeking should swing the conservatism index")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--output", "-o")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="validate estimator math, no model calls")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    if args.backend == "openai" and args.model is None:
        args.model = httpx.get(f"{args.url.rstrip('/')}/v1/models", timeout=10).json()["data"][0]["id"]
    if args.backend == "claude" and args.model is None:
        args.model = "opus"
    args.system = PERSONAS[args.persona] + NEUTRAL_SYS

    modes = ["tcn", "hl", "grable", "framing"] if args.mode == "all" else [args.mode]
    report = {"backend": args.backend, "model": args.model, "persona": args.persona,
              "samples": args.samples, "scale": args.scale}

    if not args.score_only:
        print(f"Risk-persona battery | backend={args.backend} model={args.model} "
              f"persona={args.persona} samples={args.samples}")
        print(f"modes: {', '.join(modes)}\n")

    if "tcn" in modes:
        if not args.score_only:
            print("[tcn] prospect-theory lottery series (sigma, lambda, alpha)...")
        report["tcn"] = run_tcn(args, args.scale, args.samples)
    if "hl" in modes:
        if not args.score_only:
            print("[hl] Holt-Laury price list (CRRA r)...")
        report["hl"] = run_hl(args, args.scale, args.samples)
    if "grable" in modes:
        if not args.score_only:
            print("[grable] Grable-Lytton stated risk tolerance...")
        report["grable"] = run_grable(args, args.samples)
    if "framing" in modes:
        if not args.score_only:
            print("[framing] reflection effect + ambiguity aversion...")
        report["framing"] = run_framing(args, args.samples)

    sigma = report.get("tcn", {}).get("sigma")
    lam = report.get("tcn", {}).get("lambda")
    crra = report.get("hl", {}).get("crra_r")
    composite = composite_conservatism(sigma, lam, crra)
    report["conservatism_index"] = composite

    if args.score_only:
        print(composite if composite is not None else "NA")
    else:
        print_report(report, sigma, lam, report.get("tcn", {}).get("alpha"), crra, composite)

    if args.output:
        path = args.output if os.path.isabs(args.output) else os.path.join(BASE, args.output)
        json.dump(report, open(path, "w"), indent=2)
        if not args.score_only:
            print(f"\nReport: {path}")


def print_report(report, sigma, lam, alpha, crra, composite):
    print("\n" + "=" * 72)
    print("  FINANCIAL RISK-PERSONA PROFILE")
    print("=" * 72)
    print(f"  Model : {report.get('model', 'unknown')}")
    tcn = report.get("tcn", {})
    if "error" in tcn:
        print(f"  prospect theory : ERROR — {tcn['error']}")
    elif sigma is not None:
        print(f"  sigma (curvature)     : {sigma:+.3f}   {band_sigma(sigma)}")
        print(f"     (series1 {tcn['sigma_series1']:+.2f} / series2 {tcn['sigma_series2']:+.2f})")
        print(f"  lambda (loss aversion): {lam:+.3f}   {band_lambda(lam)}")
        print(f"  alpha (prob weighting): {alpha:+.3f}   {band_alpha(alpha)}")
    hl = report.get("hl", {})
    if hl:
        if "error" in hl:
            print(f"  Holt-Laury CRRA r     : ERROR — {hl['error']}")
        else:
            print(f"  Holt-Laury CRRA r     : {crra:+.3f}   {band_sigma(crra)}  "
                  f"({hl['safe_choices']} safe choices)")
    gr = report.get("grable", {})
    if gr:
        if "error" in gr:
            print(f"  Grable-Lytton (stated): n/a — {gr['error']}")
        else:
            print(f"  Grable-Lytton (stated): {gr['grable_score']}  -> {gr['category']}")
    fr = report.get("framing", {})
    if fr and "error" not in fr:
        print(f"  reflection effect     : {fr['reflection_effect']}  "
              f"(>0 = risk-seeking in losses / framing-sensitive)")
        print(f"  ambiguity aversion    : {fr['ambiguity_aversion']}  "
              f"(>0 = shuns ambiguous odds)")
    print("-" * 72)
    if composite is not None:
        bar = round(composite / 100 * 50)
        print(f"  CONSERVATISM INDEX : {composite:3d} / 100   ({conservatism_label(composite)})")
        print(f"  risk-liberal 0 [{'-'*bar}|{'-'*(50-bar)}] 100 conservative")
    else:
        print("  CONSERVATISM INDEX : NA (no revealed-preference axis measured)")
    print("=" * 72)
    if sigma is not None and gr and "error" not in gr:
        print("  Note: compare REVEALED (sigma/lambda) vs STATED (Grable) — divergence is a finding.")
    errs = sum(v.get("errors", 0) for v in (tcn, hl, gr, fr) if isinstance(v, dict))
    print(f"  total call errors: {errs}")
    print("=" * 72)


if __name__ == "__main__":
    main()
