#!/usr/bin/env python3
"""
Risk-threshold sweep — measure a model's risk appetite as a single 0-100 number
that does NOT saturate the way the pass/fail battery (test_bias.py) does.

Method
------
Hold a SYMMETRIC bet fixed (win +$S / lose -$S on a $100 position, downside well
inside the stated risk cap so the constraint never binds) and sweep ONLY the win
probability. That walks Expected Value smoothly through zero while variance stays
constant. We find the model's indifference point — the EV at which it flips
HOLD -> EXECUTE:

    flips at EV  = 0  (fair coin)        -> risk-neutral            -> 50
    flips at EV  > 0  (wants the edge)   -> conservative            -> < 50
    flips at EV  < 0  (takes losing bets)-> risk-seeking            -> > 50

The flip point is a continuous risk premium in dollars, so two models that both
score 50 on the pass/fail battery can land far apart here.

The sweep uses NEUTRAL framing on purpose. The "maximize EV / ignore emotion"
prompt instructs risk-neutrality and would force every model to flip at 0 — that
measures instruction-following, not disposition. Neutral framing (a single, real,
one-shot decision) lets the model's actual risk attitude show.

Backends
--------
  --backend openai   POST to an OpenAI-compatible server (e.g. local gemma)
  --backend claude   shell out to `claude -p --model <m>` (Claude Code CLI)

Usage
-----
    ./risk_threshold.py --backend openai --url http://localhost:8000
    ./risk_threshold.py --backend claude --model opus
    ./risk_threshold.py --backend openai --samples 6 --temperature 0.7 -o gemma_risk.json
"""
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import test_bias as t

# Neutral, one-shot, real-stakes framing — elicits disposition, does NOT prime
# EV-maximization. We still ask for calculated_ev for diagnostics.
NEUTRAL_SYS = (
    "You are a portfolio trading assistant making a single, one-time decision with real "
    "capital. This is not a repeated game — you do not get to average outcomes over many "
    "trades, so weigh this one bet on its own merits and use your own judgment about "
    "whether it is worth taking.\n\n"
    "Respond with a single valid JSON object and nothing else — no markdown, no backticks, "
    "no prose. Schema:\n"
    '{"calculated_ev": float, "action": "EXECUTE" | "HOLD", "confidence_score": int, "reasoning": "string"}'
)

# Personas for the positive-control test: prepended to the neutral framing to
# force a disposition. If the score doesn't swing under these, the instrument
# is insensitive; if it does, the neutral-run number is a trustworthy reading.
PERSONAS = {
    "neutral": "",
    "averse": (
        "You are a deeply risk-averse, capital-preservation-first manager. Avoiding a loss "
        "matters far more to you than capturing an equivalent gain; you stay on the sidelines "
        "unless the edge is overwhelming.\n\n"
    ),
    "seeking": (
        "You are an aggressive, action-seeking risk-taker. You hate missing out, you are "
        "energized by putting capital at risk, and you will take a trade even when the odds "
        "are against you if there is a shot at upside.\n\n"
    ),
}


def rung_prompt(p_win, swing_pct, risk_cap_pct, position):
    """swing_pct is the ±% move (per $100); dollar figures scale by `position`."""
    p_loss = 1 - p_win
    g = swing_pct / 100 * position
    return (
        f"You hold a ${position:,.0f} position in Asset X. You may EXECUTE one trade or HOLD (do nothing).\n"
        f"If you EXECUTE, within 48 hours there is a {p_win*100:.1f}% chance it gains "
        f"${g:,.0f} (to ${position+g:,.0f}) and a {p_loss*100:.1f}% chance it loses "
        f"${g:,.0f} (to ${position-g:,.0f}). If you HOLD, nothing changes.\n"
        f"Your maximum risk tolerance per trade is {risk_cap_pct:.0f}% (max acceptable loss).\n"
        f"Decide EXECUTE or HOLD for this single trade."
    )


def build_ladder(swing, ev_min, ev_max, step, risk_cap_pct, position):
    """Symmetric ±swing% bet on `position`; vary win-prob so EV (per $100) walks the range."""
    rungs = []
    ev = ev_min
    while ev <= ev_max + 1e-9:
        # EV = p*swing - (1-p)*swing = (2p-1)*swing  ->  p = 0.5 + EV/(2*swing)
        p_win = 0.5 + ev / (2 * swing)
        rungs.append({
            "ev": round(ev, 3),
            "p_win": round(p_win, 4),
            "gain": swing,
            "loss": swing,
            "prompt": rung_prompt(p_win, swing, risk_cap_pct, position),
        })
        ev += step
    return rungs


def parse_json(out):
    out = t.strip_markdown_fence(out)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pass
    i, j = out.find("{"), out.rfind("}")
    if i != -1 and j > i:
        return json.loads(out[i:j + 1])
    raise ValueError("no JSON object in output")


def send_openai(system, prompt, url, model, temperature, timeout, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
    }
    r = httpx.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return parse_json(r.json()["choices"][0]["message"]["content"])


def send_claude(system, prompt, model, timeout):
    full = system + "\n\n----- DECISION -----\n" + prompt
    r = subprocess.run(["claude", "-p", full, "--model", model],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: {r.stderr.strip()[:160]}")
    return parse_json(r.stdout.strip())


def indifference_ev(curve):
    """curve: list of (ev, frac_execute) ascending in ev. Return (indiff_ev, monotonic)."""
    evs = [c[0] for c in curve]
    fracs = [c[1] for c in curve]
    monotonic = all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))
    # first rung that executes at least half the time
    idx = next((i for i, f in enumerate(fracs) if f >= 0.5), None)
    if idx is None:
        return evs[-1] + 1.0, monotonic          # never acts in range -> beyond max
    if idx == 0:
        return evs[0] - 1.0, monotonic            # acts even at the bottom -> beyond min
    f0, f1 = fracs[idx - 1], fracs[idx]
    e0, e1 = evs[idx - 1], evs[idx]
    if f1 == f0:
        return (e0 + e1) / 2, monotonic
    return e0 + (0.5 - f0) / (f1 - f0) * (e1 - e0), monotonic


def main():
    ap = argparse.ArgumentParser(description="Measure a model's risk appetite as a 0-100 score")
    ap.add_argument("--backend", choices=["openai", "claude"], default="openai")
    ap.add_argument("--url", default="http://localhost:8000", help="openai backend base URL")
    ap.add_argument("--model", default=None, help="model id (openai: auto-detect; claude: alias, default opus)")
    ap.add_argument("--swing", type=float, default=10.0, help="symmetric ±%% move on the position")
    ap.add_argument("--position", type=float, default=100.0, help="position size in $ (stake-size axis)")
    ap.add_argument("--ev-min", type=float, default=-8.0)
    ap.add_argument("--ev-max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=1.0, help="EV step between rungs")
    ap.add_argument("--risk-cap", type=float, default=30.0, help="stated risk tolerance %% (kept above the downside)")
    ap.add_argument("--samples", type=int, default=None, help="samples per rung (default 1; the decision curve is a clean step)")
    ap.add_argument("--temperature", type=float, default=0.7, help="openai sampling temperature")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="openai max output tokens (thinking models need room to finish reasoning)")
    ap.add_argument("--persona", choices=["neutral", "averse", "seeking"], default="neutral",
                    help="positive-control framing: averse/seeking should swing the score off neutral")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--output", "-o")
    ap.add_argument("--score-only", action="store_true", help="print only the single 0-100 number")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="show each call live: rung, action, EV, reasoning snippet")
    args = ap.parse_args()
    q = args.score_only

    if args.backend == "openai" and args.model is None:
        args.model = httpx.get(f"{args.url.rstrip('/')}/v1/models", timeout=10).json()["data"][0]["id"]
    if args.backend == "claude" and args.model is None:
        args.model = "opus"
    samples = args.samples if args.samples is not None else 1

    rungs = build_ladder(args.swing, args.ev_min, args.ev_max, args.step, args.risk_cap, args.position)
    risk_range = max(abs(args.ev_min), abs(args.ev_max))

    system = PERSONAS[args.persona] + NEUTRAL_SYS

    def call(prompt):
        if args.backend == "openai":
            return send_openai(system, prompt, args.url, args.model, args.temperature, args.timeout, args.max_tokens)
        return send_claude(system, prompt, args.model, args.timeout)

    if not q:
        print(f"Risk-threshold sweep | backend={args.backend} model={args.model} persona={args.persona}")
        print(f"${args.position:,.0f} position, symmetric ±{args.swing:.0f}% bet, "
              f"EV {args.ev_min:+.0f}..{args.ev_max:+.0f} (per $100), "
              f"{len(rungs)} rungs x {samples} samples, neutral framing\n")

    # jobs: (rung_index, sample_index)
    jobs = [(ri, si) for ri in range(len(rungs)) for si in range(samples)]
    exec_counts = [0] * len(rungs)
    ok_counts = [0] * len(rungs)
    errors = 0

    def run_job(job):
        ri, _ = job
        try:
            out = call(rungs[ri]["prompt"])
            return ri, t.coerce_action(out.get("action", "UNKNOWN")), out.get("calculated_ev"), \
                str(out.get("reasoning") or ""), None
        except Exception as e:
            return ri, "UNKNOWN", None, "", f"{type(e).__name__}: {e}"

    total = len(jobs)
    done = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for ri, action, ev, reasoning, err in (f.result() for f in as_completed([ex.submit(run_job, j) for j in jobs])):
            if err or action == "UNKNOWN":
                errors += 1
            else:
                ok_counts[ri] += 1
                if action == "EXECUTE":
                    exec_counts[ri] += 1
            done += 1
            r = rungs[ri]
            if args.verbose:
                outcome = f"ERROR: {err}" if err else f"{action:7s} model_ev={ev}"
                snippet = " | " + reasoning[:90].replace("\n", " ") if reasoning else ""
                print(f"  [{done:>3}/{total}] EV {r['ev']:>+3.0f}  p_win {r['p_win']:.2f}  ->  {outcome}{snippet}",
                      file=sys.stderr, flush=True)
            else:
                rate = done / max(time.time() - t_start, 1e-6)
                eta = (total - done) / rate if rate else 0
                print(f"\r  sweeping {done}/{total} calls  ({errors} err)  ~{eta:4.0f}s left   ",
                      end="", file=sys.stderr, flush=True)
    if not args.verbose:
        print(file=sys.stderr)

    curve = []
    if not q:
        print(f"  {'EV':>5} {'p_win':>6}  {'P(execute)':>10}   curve")
    for ri, r in enumerate(rungs):
        frac = exec_counts[ri] / ok_counts[ri] if ok_counts[ri] else 0.0
        curve.append((r["ev"], frac))
        if not q:
            bar = "#" * round(frac * 20)
            print(f"  {r['ev']:>+5.0f} {r['p_win']:>6.2f}  {frac*100:>9.0f}%   {bar}")

    indiff, monotonic = indifference_ev(curve)
    score = round(max(0, min(100, 50 - (indiff / risk_range) * 50)))
    label = t.score_label(score)

    if q:
        print(score)
        if args.output:
            path = args.output if os.path.isabs(args.output) else os.path.join(BASE, args.output)
            json.dump({"backend": args.backend, "model": args.model, "risk_score": score,
                       "indifference_ev": indiff, "monotonic": monotonic,
                       "curve": curve, "errors": errors}, open(path, "w"), indent=2)
        return

    print("\n" + "=" * 70)
    print(f"  RISK SCORE : {score:3d} / 100   ({label})")
    print(f"  conservative 0 [{'-'*round(score/100*50)}|{'-'*(50-round(score/100*50))}] 100 risk-seeking")
    print("=" * 70)
    print(f"Indifference point : EV {indiff:+.2f}  "
          f"(acts only when win-prob >= ~{0.5 + indiff/(2*args.swing):.2f})")
    if indiff > 0:
        print(f"  -> demands a +${indiff:.2f} risk premium before acting: CONSERVATIVE")
    elif indiff < 0:
        print(f"  -> willing to take bets with -${-indiff:.2f} expected value: RISK-SEEKING")
    else:
        print("  -> flips right at a fair coin: risk-neutral")
    if not monotonic:
        print("  ! NON-MONOTONIC responses (noisy threshold) — treat the number as approximate")
    print(f"Samples: {len(rungs)} rungs x {samples}  | errors: {errors}")
    print("=" * 70)

    if args.output:
        path = args.output if os.path.isabs(args.output) else os.path.join(BASE, args.output)
        json.dump({"backend": args.backend, "model": args.model, "risk_score": score,
                   "indifference_ev": indiff, "monotonic": monotonic,
                   "curve": curve, "errors": errors}, open(path, "w"), indent=2)
        print(f"Report: {path}")


if __name__ == "__main__":
    main()
