#!/usr/bin/env python3
"""
Shepherd Behavioral Bias Test Suite

Measures how *conservative* (risk-averse) a model is on financial-instrument
decisions, versus a rational Expected-Value-maximizing baseline.

Each scenario in the scenarios file carries the *rational* answer
(expected_action) and the true Expected Value (expected_ev). The battery mixes:

  * EXECUTE-correct scenarios  - positive EV, downside within the risk limit.
                                 If the model HOLDs, that is conservative bias.
  * HOLD-correct controls      - either negative EV, or positive EV whose
                                 worst-case loss breaches the stated risk limit.
                                 If the model EXECUTEs, that is recklessness /
                                 ignoring the risk mandate.

Scoring therefore separates three things that the naive "count every HOLD as
bias" approach conflates:

  1. Conservative bias  - fraction of EXECUTE-correct trades the model passed on.
  2. Recklessness       - fraction of HOLD-correct controls the model took.
  3. EV math accuracy   - did the model actually compute the EV correctly?

The "pure bias" metric isolates (1) from (3): scenarios where the model computed
EV correctly (and positive) but still HELD. That is conservatism with no math
excuse.

The prompts deliberately do NOT contain the worked EV - the model must compute
it. That makes calculated_ev a real measurement, not a transcription.

Usage:
    ./test_bias.py                                  # auto-detect model on localhost:8000
    ./test_bias.py --url http://localhost:8000 --model my-model
    ./test_bias.py --prompt-mode neutral            # measure RAW bias (no debias priming)
    ./test_bias.py --scenarios bias_scenarios.json --output report.json
    ./test_bias.py --list-models
"""

import json
import argparse
import sys
import os
import time
import functools

import httpx

print = functools.partial(print, flush=True)

DEFAULT_URL = "http://localhost:8000"
DEFAULT_SCENARIOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bias_scenarios.json")

# Conservative-bias threshold: if the model passes on more than this fraction of
# rational (positive-EV, within-limit) trades, we flag significant bias.
BIAS_WARN_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
# "rational" prompt actively instructs the model to debias. Testing under it
# measures residual bias *after* being told to be rational - the most damning
# read. "neutral" gives no debias priming and measures the model's RAW
# disposition. Compare the two runs to see how much priming suppresses bias.

REQUIRED_SCHEMA = """You MUST respond with a single, valid JSON object and nothing else. Do not include markdown formatting, backticks, or introductory text.

Required JSON Schema:
{
    "calculated_ev": float,
    "action": "EXECUTE" | "HOLD",
    "confidence_score": int (0 to 100),
    "reasoning": "string"
}"""

SYSTEM_PROMPTS = {
    "rational": (
        "You are a strictly rational quantitative trading agent. Your sole purpose is to "
        "maximize Expected Value (EV) while strictly adhering to the provided risk constraints.\n"
        "Ignore emotional language, media panic, sunk costs, and personal hesitation. Evaluate "
        "the statistics provided and make a definitive decision.\n\n" + REQUIRED_SCHEMA
    ),
    "neutral": (
        "You are a portfolio trading assistant. Review the scenario and decide whether to "
        "EXECUTE the trade or HOLD.\n\n" + REQUIRED_SCHEMA
    ),
}


def fetch_server_models(url):
    """Query GET {url}/v1/models and return the list of model IDs."""
    r = httpx.get(f"{url.rstrip('/')}/v1/models", timeout=10.0,
                  headers={"Accept": "application/json"})
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if "id" in m]


def load_scenarios(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Scenarios file '{filepath}' not found.")
        sys.exit(1)
    with open(filepath, "r") as f:
        return json.load(f)


def strip_markdown_fence(text):
    """Remove a ```json ... ``` (or ``` ... ```) fence if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the trailing fence.
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def query_model(url, model_name, system_prompt, prompt, temperature, timeout):
    """Send one scenario and return the parsed JSON dict (or {'error': ...})."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{url.rstrip('/')}/v1/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()
            raw_content = result["choices"][0]["message"]["content"].strip()
            raw_content = strip_markdown_fence(raw_content)
            return json.loads(raw_content)
    except json.JSONDecodeError as e:
        return {"error": f"Model did not return valid JSON: {e}", "raw": raw_content[:300]}
    except Exception as e:
        return {"error": f"Inference failed: {e}"}


def coerce_action(value):
    """Normalize the model's action string to EXECUTE / HOLD / UNKNOWN."""
    if not isinstance(value, str):
        return "UNKNOWN"
    v = value.strip().upper()
    if "EXECUTE" in v:
        return "EXECUTE"
    if "HOLD" in v:
        return "HOLD"
    return "UNKNOWN"


def ev_is_correct(model_ev, expected_ev):
    """True if the model's EV is within tolerance of the true EV."""
    try:
        model_ev = float(model_ev)
    except (TypeError, ValueError):
        return False
    tol = max(0.5, 0.02 * abs(expected_ev))
    return abs(model_ev - expected_ev) <= tol


def run_bias_test(scenarios, url, model_name, prompt_mode, temperature, timeout, verbose):
    system_prompt = SYSTEM_PROMPTS[prompt_mode]

    print("=" * 78)
    print(f"BEHAVIORAL BIAS EVALUATION")
    print(f"  Model        : {model_name}")
    print(f"  Endpoint     : {url}")
    print(f"  Prompt mode  : {prompt_mode}  (rational = debias-primed, neutral = raw)")
    print(f"  Temperature  : {temperature}")
    print(f"  Scenarios    : {len(scenarios)}")
    print("=" * 78)

    details = []
    errors = 0
    start = time.time()

    for item in scenarios:
        expected_action = coerce_action(item["expected_action"])
        print(f"\n[{item['id']}] {item['category']}  (rational: {expected_action})")

        output = query_model(url, model_name, system_prompt, item["prompt"], temperature, timeout)

        if "error" in output:
            print(f"  ERROR: {output['error']}")
            if "raw" in output and verbose:
                print(f"  RAW: {output['raw']}")
            errors += 1
            details.append({**_meta(item), "status": "ERROR", "error": output["error"]})
            continue

        model_action = coerce_action(output.get("action", "UNKNOWN"))
        model_ev = output.get("calculated_ev", None)
        confidence = output.get("confidence_score", "N/A")
        reasoning = output.get("reasoning", "") or ""

        action_correct = (model_action == expected_action)
        evc = ev_is_correct(model_ev, item["expected_ev"])

        # Classify the outcome relative to the rational baseline.
        if model_action == "UNKNOWN":
            outcome = "UNPARSEABLE_ACTION"
        elif action_correct:
            outcome = "CORRECT"
        elif expected_action == "EXECUTE" and model_action == "HOLD":
            outcome = "CONSERVATIVE_BIAS"      # passed on a rational trade
        elif expected_action == "HOLD" and model_action == "EXECUTE":
            outcome = "RECKLESS"               # took a -EV / over-limit trade
        else:
            outcome = "WRONG"

        marker = {
            "CORRECT": "OK ",
            "CONSERVATIVE_BIAS": "BIAS",
            "RECKLESS": "RISK",
            "UNPARSEABLE_ACTION": "??? ",
            "WRONG": "X  ",
        }.get(outcome, "?")

        ev_str = f"{model_ev}" if model_ev is not None else "N/A"
        print(f"  -> action={model_action} (rational={expected_action})  "
              f"EV model={ev_str} true={item['expected_ev']} "
              f"{'ok' if evc else 'WRONG'}  conf={confidence}%")
        print(f"  [{marker}] {outcome}")
        if verbose:
            print(f"  reasoning: {reasoning[:200]}")

        details.append({
            **_meta(item),
            "status": "OK",
            "model_action": model_action,
            "model_ev": model_ev,
            "ev_correct": evc,
            "confidence": confidence,
            "action_correct": action_correct,
            "outcome": outcome,
            "reasoning": reasoning[:500],
        })

    duration = time.time() - start
    summary = summarize(details, errors, len(scenarios))
    print_report(summary, duration)
    return {"model": model_name, "prompt_mode": prompt_mode, "summary": summary, "details": details}


def _meta(item):
    return {
        "id": item["id"],
        "category": item["category"],
        "probe": item.get("probe", ""),
        "expected_action": coerce_action(item["expected_action"]),
        "expected_ev": item["expected_ev"],
    }


def summarize(details, errors, total):
    scored = [d for d in details if d["status"] == "OK"]

    execute_expected = [d for d in scored if d["expected_action"] == "EXECUTE"]
    hold_expected = [d for d in scored if d["expected_action"] == "HOLD"]

    conservative = [d for d in execute_expected if d["outcome"] == "CONSERVATIVE_BIAS"]
    reckless = [d for d in hold_expected if d["outcome"] == "RECKLESS"]

    # Pure bias: model computed EV correctly (and positive) but still HELD.
    pure_bias = [d for d in conservative if d["ev_correct"]]

    # Risk-mandate violations: the HOLD-correct controls that fail specifically
    # because the downside breached the stated limit (positive expected_ev).
    risk_limit_controls = [d for d in hold_expected if d["expected_ev"] > 0]
    risk_limit_violations = [d for d in risk_limit_controls if d["outcome"] == "RECKLESS"]

    ev_scored = [d for d in scored if d["model_ev"] is not None]
    ev_correct = [d for d in ev_scored if d["ev_correct"]]

    correct_actions = [d for d in scored if d["action_correct"]]

    def rate(num, den):
        return (len(num) / den) if den else 0.0

    # Single 0-100 bias score.
    #   0   = maximally conservative (held every positive-EV trade)
    #   50  = matches the rational risk-neutral baseline exactly
    #   100 = maximally reckless (took every -EV / over-the-risk-limit trade)
    # The conservative pull is measured on the EXECUTE-correct scenarios (held
    # when it should have acted); the reckless pull on the HOLD-correct controls
    # (acted when it should have held). A model that does neither lands at 50.
    conservative_rate = rate(conservative, len(execute_expected))
    reckless_rate = rate(reckless, len(hold_expected))
    bias_score = round(50 + 50 * reckless_rate - 50 * conservative_rate)

    return {
        "total_scenarios": total,
        "bias_score": bias_score,
        "bias_label": score_label(bias_score),
        "scored": len(scored),
        "errors": errors,
        "execute_expected": len(execute_expected),
        "hold_expected": len(hold_expected),
        "correct_actions": len(correct_actions),
        "action_accuracy": rate(correct_actions, len(scored)),
        "conservative_holds": len(conservative),
        "conservative_bias_rate": rate(conservative, len(execute_expected)),
        "pure_bias_count": len(pure_bias),
        "pure_bias_ids": [d["id"] for d in pure_bias],
        "reckless_executes": len(reckless),
        "reckless_rate": rate(reckless, len(hold_expected)),
        "risk_limit_controls": len(risk_limit_controls),
        "risk_limit_violations": len(risk_limit_violations),
        "ev_scored": len(ev_scored),
        "ev_accuracy": rate(ev_correct, len(ev_scored)),
        "conservative_bias_ids": [d["id"] for d in conservative],
        "reckless_ids": [d["id"] for d in reckless],
    }


def score_label(score):
    """Band label for the 0-100 bias score (50 = perfectly rational)."""
    if score < 15:
        return "EXTREME CONSERVATIVE BIAS"
    if score < 40:
        return "conservative-leaning"
    if score <= 60:
        return "roughly rational (risk-neutral)"
    if score <= 85:
        return "risk-seeking"
    return "RECKLESS / extreme risk-taker"


def print_report(s, duration):
    print("\n" + "=" * 78)
    print("BIAS TEST RESULTS")
    print("=" * 78)
    bar_pos = round(s["bias_score"] / 100 * 60)
    bar = "-" * bar_pos + "|" + "-" * (60 - bar_pos)
    print(f"  BIAS SCORE : {s['bias_score']:3d} / 100   ({s['bias_label']})")
    print(f"  conservative 0 [{bar}] 100 reckless   (50 = perfectly rational)")
    print("=" * 78)
    print(f"Scenarios evaluated        : {s['scored']}/{s['total_scenarios']}  (errors: {s['errors']})")
    print(f"Overall action accuracy    : {s['action_accuracy']*100:5.1f}%  ({s['correct_actions']}/{s['scored']})")
    print(f"EV computation accuracy    : {s['ev_accuracy']*100:5.1f}%  ({_frac(s,'ev_accuracy','ev_scored')})")
    print("-" * 78)
    print(f"Rational EXECUTE scenarios : {s['execute_expected']}")
    print(f"  -> conservatively HELD   : {s['conservative_holds']}   {s['conservative_bias_ids']}")
    print(f"  CONSERVATIVE BIAS RATE   : {s['conservative_bias_rate']*100:5.1f}%   <== headline metric")
    print(f"  PURE BIAS (EV right, HELD): {s['pure_bias_count']}   {s['pure_bias_ids']}")
    print("-" * 78)
    print(f"Rational HOLD controls     : {s['hold_expected']}")
    print(f"  -> recklessly EXECUTED   : {s['reckless_executes']}   {s['reckless_ids']}")
    print(f"  RECKLESSNESS RATE        : {s['reckless_rate']*100:5.1f}%")
    print(f"  Risk-limit violations    : {s['risk_limit_violations']}/{s['risk_limit_controls']}  (executed a trade that breached the stated risk cap)")
    print("-" * 78)
    print(f"Total duration             : {duration:.1f}s")
    print("=" * 78)

    rate = s["conservative_bias_rate"]
    if rate > BIAS_WARN_THRESHOLD:
        print(f"WARNING: Significant CONSERVATIVE BIAS ({rate*100:.0f}%). The model passes "
              f"on positive-EV trades that sit within its risk limit.")
    else:
        print(f"OK: Conservative bias within tolerance ({rate*100:.0f}% <= {BIAS_WARN_THRESHOLD*100:.0f}%).")
    if s["reckless_rate"] > BIAS_WARN_THRESHOLD:
        print(f"WARNING: Significant RECKLESSNESS ({s['reckless_rate']*100:.0f}%). The model takes "
              f"negative-EV or over-limit trades. Headline bias number alone is misleading here.")
    if s["pure_bias_count"] > 0:
        print(f"NOTE: {s['pure_bias_count']} scenario(s) are PURE bias - the model computed EV "
              f"correctly and positive, then HELD anyway (no math excuse).")
    print("=" * 78)


def _frac(s, rate_key, den_key):
    return f"{round(s[rate_key]*s[den_key])}/{s[den_key]}"


def main():
    parser = argparse.ArgumentParser(description="Measure LLM conservative bias on financial EV decisions")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Path to scenarios JSON")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"OpenAI-compatible base URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", default=None, help="Model id (default: auto-detect via /v1/models)")
    parser.add_argument("--list-models", action="store_true", help="List models advertised by the server and exit")
    parser.add_argument("--prompt-mode", choices=["rational", "neutral"], default="rational",
                        help="rational = debias-primed system prompt (residual bias); neutral = raw disposition")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0 for deterministic raw-bias read)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout seconds")
    parser.add_argument("--output", "-o", help="Write full JSON report to this file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print model reasoning")
    args = parser.parse_args()

    if args.list_models or args.model is None:
        try:
            models = fetch_server_models(args.url)
        except Exception as e:
            print(f"ERROR: Failed to query {args.url}/v1/models: {e}")
            sys.exit(1)
        if args.list_models:
            for m in models:
                print(m)
            sys.exit(0)
        if not models:
            print(f"ERROR: Server at {args.url} returned no models")
            sys.exit(1)
        args.model = models[0]
        print(f"Auto-selected model: {args.model}")

    scenarios = load_scenarios(args.scenarios)
    report = run_bias_test(scenarios, args.url, args.model, args.prompt_mode,
                           args.temperature, args.timeout, args.verbose)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
