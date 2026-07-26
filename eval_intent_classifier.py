"""
eval_intent_classifier.py
──────────────────────────────────────────────
Real classification_report (precision / recall / F1 per intent) for the
MuRIL (India/Hinglish) and BERT (Global/English) intent classifiers used
in app.py — replacing the hardcoded "80.5% / 83.17% accuracy" numbers
shown in the UI's Intelligence Stack panel with a reproducible number.

WHY THIS EXISTS:
  Those two percentages were hardcoded directly into the Gradio HTML with
  no eval script or labeled test set checked into this repo, so there was
  no way to reproduce them or know what "accuracy" was measured against.
  This script gives a real, versioned eval set and a real metric.

WHAT'S IN eval_set.json:
  A small hand-labeled starter set (~6-8 examples per intent, per market).
  This is enough to sanity-check the classifier and catch obvious
  regressions, but it is NOT a substitute for a large eval set drawn from
  real customer messages — treat these F1 numbers as a floor, not a final
  score. Grow eval_set.json from real (anonymized) WhatsApp/chat logs as
  they accumulate for a trustworthy number.

REQUIREMENTS TO RUN:
  This needs the actual MuRIL/BERT model weights + index files that app.py
  loads from models/muril_finetuned, models/bert_finetuned, and index/*.npy
  — the same files app.py itself needs and which are NOT checked into this
  repo (too large for git / pulled from HuggingFace separately). It cannot
  be run in an environment that doesn't already run app.py successfully.

USAGE:
  python eval_intent_classifier.py                # both markets
  python eval_intent_classifier.py --market india  # MuRIL only
  python eval_intent_classifier.py --market global # BERT only
"""
import argparse
import json
import sys
from pathlib import Path

from sklearn.metrics import classification_report

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"


def load_eval_set(market: str) -> list[dict]:
    with open(EVAL_SET_PATH) as f:
        data = json.load(f)
    return data[market]


def run_eval(agent, market: str) -> None:
    examples = load_eval_set(market)
    y_true = [ex["label"] for ex in examples]
    y_pred = []

    for ex in examples:
        pred_label, confidence = agent.classify_intent(ex["text"], market)
        y_pred.append(pred_label)

    print(f"\n{'=' * 60}")
    print(f"  {market.upper()} classifier — {len(examples)} labeled examples")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, zero_division=0))

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    print(f"Accuracy: {correct}/{len(y_true)} = {correct / len(y_true):.1%}")

    mismatches = [
        (ex["text"], ex["label"], pred)
        for ex, pred in zip(examples, y_pred)
        if ex["label"] != pred
    ]
    if mismatches:
        print(f"\n{len(mismatches)} misclassified:")
        for text, true_label, pred_label in mismatches:
            print(f"  [{true_label} -> {pred_label}] {text!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["india", "global", "both"], default="both")
    args = parser.parse_args()

    # Imported lazily — this pulls in torch/transformers and loads the real
    # model weights, so only do it once we know eval_set.json exists.
    if not EVAL_SET_PATH.exists():
        sys.exit(f"Missing {EVAL_SET_PATH} — run this script from the repo root.")

    from app import BargainingAgent
    print("Loading agent (MuRIL + BERT)...")
    agent = BargainingAgent()

    markets = ["india", "global"] if args.market == "both" else [args.market]
    for market in markets:
        run_eval(agent, market)


if __name__ == "__main__":
    main()
