"""Run all approaches on the same test tickets and write a side-by-side comparison.

Examples:
    python -m tickets.compare --methods baseline               # free, instant
    python -m tickets.compare --methods baseline,finetune      # after `python -m tickets.finetune`
    python -m tickets.compare --limit 50 --methods llm         # small, cheap first LLM run
    python -m tickets.compare                                   # everything
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tickets import plots
from tickets.baseline import TfidfClassifier
from tickets.config import (
    FINETUNED_DIR,
    LABELS,
    RANDOM_STATE,
    REPORTS_DIR,
    SHOTS_PER_CLASS,
)
from tickets.data import load_datasets
from tickets.evaluate import evaluate, llm_cost_per_1000
from tickets.llm import LLMClassifier, build_system_prompt, make_client, select_few_shot
from tickets.report import write_report

BASELINE_NAME = "TF-IDF + LR (baseline)"
FINETUNE_NAME = "Fine-tuned DistilBERT"


def _save_predictions(test: pd.DataFrame, preds, path: Path) -> None:
    out = test.copy()
    out["predicted"] = [p.label for p in preds]
    out["correct"] = out["label"] == out["predicted"]
    out["latency_ms"] = [round(p.latency_s * 1000, 2) for p in preds]
    if any(p.input_tokens for p in preds):
        out["input_tokens"] = [p.input_tokens for p in preds]
        out["output_tokens"] = [p.output_tokens for p in preds]
        out["raw_reply"] = [p.raw for p in preds]
    out.to_csv(path, index=False)


def run_comparison(
    methods: list[str],
    data: dict[str, pd.DataFrame] | None = None,
    reports_dir: Path = REPORTS_DIR,
    finetuned_dir: Path = FINETUNED_DIR,
    limit: int | None = None,
    shots: int = SHOTS_PER_CLASS,
    llm_client=None,
    verbose: bool = True,
) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    data = data or load_datasets()
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    test = data["test"]
    if limit and limit < len(test):
        test = test.sample(n=limit, random_state=RANDOM_STATE).reset_index(drop=True)
    texts = test["text"].tolist()
    log(f"Evaluating on {len(test)} test tickets\n")

    results: dict[str, dict] = {}
    extras: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    predictions = {}

    if "baseline" in methods:
        log("Running TF-IDF baseline...")
        clf = TfidfClassifier().fit(data["train"])
        predictions[BASELINE_NAME] = ("baseline", clf.predict(texts))

    if "finetune" in methods:
        log("Running fine-tuned model...")
        try:
            from tickets.finetune import FineTunedClassifier

            clf = FineTunedClassifier(finetuned_dir)
            clf.predict(texts[:3])  # warm-up so one-off start-up cost does not skew latency
            predictions[FINETUNE_NAME] = ("finetune", clf.predict(texts))
            summary_path = Path(finetuned_dir) / "training_summary.json"
            if summary_path.exists():
                extras[FINETUNE_NAME] = {"training": json.loads(summary_path.read_text())}
        except ImportError as exc:
            skipped[FINETUNE_NAME] = f"needs `pip install torch transformers` ({exc})"
        except FileNotFoundError as exc:
            skipped[FINETUNE_NAME] = str(exc)

    llm_name = f"LLM few-shot ({shots}/class)" if shots else "LLM zero-shot"
    if "llm" in methods:
        log(f"Running LLM ({len(texts)} API calls; cached replies are free)...")
        try:
            client = llm_client or make_client()
            examples = select_few_shot(data["train"], shots) if shots else None
            prompt = build_system_prompt(examples=examples)
            llm = LLMClassifier(prompt, client=client, cache_path=reports_dir / "llm_cache.jsonl", verbose=verbose)
            preds = llm.predict(texts)
            predictions[llm_name] = ("llm", preds)
            extras[llm_name] = {"cost_per_1000": llm_cost_per_1000(preds), "prompt_chars": len(prompt)}
        except RuntimeError as exc:
            skipped[llm_name] = str(exc)

    for name, (slug, preds) in predictions.items():
        results[name] = evaluate(test, preds, LABELS)
        _save_predictions(test, preds, reports_dir / f"predictions_{slug}.csv")
        plots.plot_confusion(results[name]["confusion_matrix"], LABELS, name, figures_dir / f"confusion_{slug}.png")

    if not results:
        raise RuntimeError("Nothing ran. " + "; ".join(f"{k}: {v}" for k, v in skipped.items()))

    plots.plot_accuracy_by_wording(results, figures_dir / "accuracy_by_wording.png")
    write_report(results, extras, skipped, reports_dir / "comparison.md")
    (reports_dir / "results.json").write_text(json.dumps({"results": results, "extras": extras}, indent=2, default=float))

    log(f"{'method':<28}{'accuracy':>10}{'macro-F1':>10}{'p50 latency':>14}")
    for name, r in results.items():
        log(f"{name:<28}{r['accuracy']:>10.1%}{r['macro_f1']:>10.3f}{r['latency']['p50_ms']:>11.1f} ms")
    for name, why in skipped.items():
        log(f"SKIPPED {name}: {why}")
    log(f"\nWrote {reports_dir / 'comparison.md'}")
    return {"results": results, "extras": extras, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline, fine-tuned model and prompted LLM.")
    parser.add_argument("--methods", default="baseline,finetune,llm", help="comma-separated: baseline,finetune,llm")
    parser.add_argument("--limit", type=int, default=None, help="evaluate on a random subset of N test tickets")
    parser.add_argument("--shots", type=int, default=SHOTS_PER_CLASS, help="few-shot examples per class (0 = zero-shot)")
    args = parser.parse_args()
    run_comparison([m.strip() for m in args.methods.split(",")], limit=args.limit, shots=args.shots)


if __name__ == "__main__":
    main()
