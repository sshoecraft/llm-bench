#!/usr/bin/env python3
"""
Run the financial-bias battery through the Claude Code CLI (`claude -p`) and
score it with test_bias.py's 0-100 bias metric.

Each scenario is sent as:  <rational/neutral system framing> + <scenario prompt>
folded into a single `claude -p` call, so the JSON output is parseable and the
run is comparable to the OpenAI-backend path in test_bias.py.

Note: `claude -p` drives the full Claude Code agent (its system prompt + tools),
so this measures the model *as Claude Code*, not a raw model-API read.

Usage:
    ./run_opus_bias.py                         # opus, rational framing
    ./run_opus_bias.py --model sonnet
    ./run_opus_bias.py --prompt-mode neutral   # raw disposition (no debias priming)
    ./run_opus_bias.py --workers 4 --output opus_bias_report.json
"""
import os
import sys
import json
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import test_bias as t


def extract_json(out):
    out = t.strip_markdown_fence(out)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        pass
    i, j = out.find("{"), out.rfind("}")
    if i != -1 and j > i:
        return json.loads(out[i:j + 1])
    raise ValueError("no JSON object in output")


def classify(exp, act):
    if act == "UNKNOWN":
        return "UNPARSEABLE_ACTION"
    if act == exp:
        return "CORRECT"
    if exp == "EXECUTE" and act == "HOLD":
        return "CONSERVATIVE_BIAS"
    if exp == "HOLD" and act == "EXECUTE":
        return "RECKLESS"
    return "WRONG"


def run_one(system_prompt, model, timeout, s):
    prompt = system_prompt + "\n\n----- SCENARIO -----\n" + s["prompt"]
    try:
        r = subprocess.run(["claude", "-p", prompt, "--model", model],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return s, {"error": f"claude exit {r.returncode}: {r.stderr.strip()[:200]}"}
        return s, extract_json(r.stdout.strip())
    except Exception as e:
        return s, {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description="Score a Claude model's financial bias via `claude -p`")
    ap.add_argument("--scenarios", default=os.path.join(BASE, "bias_scenarios.json"))
    ap.add_argument("--model", default="opus", help="claude CLI model alias (default: opus)")
    ap.add_argument("--prompt-mode", choices=["rational", "neutral"], default="rational",
                    help="rational = debias-primed; neutral = raw disposition")
    ap.add_argument("--workers", type=int, default=5, help="parallel claude -p calls")
    ap.add_argument("--timeout", type=float, default=240.0, help="per-call timeout seconds")
    ap.add_argument("--output", "-o", help="write full JSON report here")
    args = ap.parse_args()

    scenarios = json.load(open(args.scenarios))
    system_prompt = t.SYSTEM_PROMPTS[args.prompt_mode]

    print(f"Running {len(scenarios)} scenarios through `claude -p --model {args.model}` "
          f"({args.prompt_mode} framing, {args.workers} parallel)...\n")
    start = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, system_prompt, args.model, args.timeout, s): s for s in scenarios}
        done = 0
        for fut in as_completed(futs):
            s, out = fut.result()
            results[s["id"]] = out
            done += 1
            if "error" in out:
                print(f"  [{done:2d}/{len(scenarios)}] {s['id']}  ERROR: {out['error']}")
            else:
                a = t.coerce_action(out.get("action", "UNKNOWN"))
                print(f"  [{done:2d}/{len(scenarios)}] {s['id']}  {a:7s} "
                      f"(rational {s['expected_action']})  EV={out.get('calculated_ev')}")

    details, errors = [], 0
    for s in scenarios:
        out = results[s["id"]]
        exp = s["expected_action"]
        if "error" in out:
            errors += 1
            details.append({"id": s["id"], "status": "ERROR",
                            "expected_action": exp, "expected_ev": s["expected_ev"]})
            continue
        act = t.coerce_action(out.get("action", "UNKNOWN"))
        details.append({
            "id": s["id"], "status": "OK", "expected_action": exp, "expected_ev": s["expected_ev"],
            "model_action": act, "model_ev": out.get("calculated_ev"),
            "ev_correct": t.ev_is_correct(out.get("calculated_ev"), s["expected_ev"]),
            "action_correct": act == exp, "outcome": classify(exp, act),
        })

    summary = t.summarize(details, errors, len(scenarios))
    t.print_report(summary, time.time() - start)

    dev = [d for d in details if d["status"] == "OK"
           and d["outcome"] in ("CONSERVATIVE_BIAS", "RECKLESS", "WRONG")]
    if dev:
        print("\nDeviations from rational baseline:")
        for d in dev:
            print(f"  {d['id']}  expected {d['expected_action']:7s} got {d['model_action']:7s}  [{d['outcome']}]")

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(BASE, args.output)
        json.dump({"model": args.model, "prompt_mode": args.prompt_mode,
                   "summary": summary, "details": details, "raw": results},
                  open(out_path, "w"), indent=2)
        print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
