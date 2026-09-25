"""Writes reports/comparison.md: a plain-language summary of the three approaches."""
from __future__ import annotations

from pathlib import Path

from tickets.config import (
    LLM_MODEL,
    LLM_PRICE_INPUT_PER_MTOK,
    LLM_PRICE_OUTPUT_PER_MTOK,
    SELF_HOST_COST_PER_MONTH,
)
from tickets.evaluate import break_even_requests_per_month


def _ms(v: float) -> str:
    return f"{v:.1f} ms" if v < 1000 else f"{v / 1000:.2f} s"


def write_report(results: dict, extras: dict, skipped: dict, path: Path) -> None:
    """results: method -> metrics from evaluate(); extras: method -> extra facts; skipped: method -> reason."""
    wordings = ["direct", "indirect", "novel"]
    header = "| Method | Accuracy | Macro-F1 | " + " | ".join(w.capitalize() for w in wordings) + " | Median latency | p95 latency |"
    divider = "|" + "---|" * (6 + len(wordings) - 1)
    rows = []
    for method, r in results.items():
        cells = [
            f"{r['by_wording'][w]['accuracy']:.1%}" if w in r["by_wording"] else "-" for w in wordings
        ]
        rows.append(
            f"| {method} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | " + " | ".join(cells)
            + f" | {_ms(r['latency']['p50_ms'])} | {_ms(r['latency']['p95_ms'])} |"
        )

    n = next(iter(results.values()))["n"]
    counts = next(iter(results.values()))["by_wording"]
    count_text = ", ".join(f"{w}: {counts[w]['n']}" for w in wordings if w in counts)

    best = max(results, key=lambda m: results[m]["accuracy"])
    hardest = max(results, key=lambda m: results[m]["by_wording"].get("novel", {"accuracy": -1})["accuracy"])
    fastest = min(results, key=lambda m: results[m]["latency"]["p50_ms"])

    findings = [
        f"- Highest overall accuracy: **{best}** ({results[best]['accuracy']:.1%}).",
        f"- Best on the hardest group (novel wording): **{hardest}** "
        f"({results[hardest]['by_wording'].get('novel', {'accuracy': float('nan')})['accuracy']:.1%}).",
        f"- Fastest (median latency): **{fastest}** ({_ms(results[fastest]['latency']['p50_ms'])}).",
    ]

    cost_lines = []
    llm_name = next((m for m in results if m.startswith("LLM")), None)
    if llm_name:
        cost = extras[llm_name]["cost_per_1000"]
        r = results[llm_name]
        cost_lines.append(
            f"- **{llm_name}** ({LLM_MODEL}): about **${cost:.3f} per 1,000 tickets** "
            f"(average {r['mean_input_tokens']:.0f} input + {r['mean_output_tokens']:.1f} output tokens per ticket, "
            f"priced at ${LLM_PRICE_INPUT_PER_MTOK:g} / ${LLM_PRICE_OUTPUT_PER_MTOK:g} per million tokens)."
        )
        if any(m.startswith("Fine-tuned") for m in results):
            be = break_even_requests_per_month(SELF_HOST_COST_PER_MONTH, cost / 1000)
            if be:
                cost_lines.append(
                    f"- **Break-even:** if hosting the fine-tuned model costs ${SELF_HOST_COST_PER_MONTH:,.0f}/month "
                    f"(an assumption, edit `SELF_HOST_COST_PER_MONTH`), self-hosting is cheaper above roughly "
                    f"**{be:,.0f} tickets per month**."
                )
    ft_name = next((m for m in results if m.startswith("Fine-tuned")), None)
    if ft_name and "training" in extras.get(ft_name, {}):
        t = extras[ft_name]["training"]
        cost_lines.append(
            f"- **{ft_name}** training took {t['train_seconds'] / 60:.1f} minutes on `{t['device']}` "
            f"(one-off cost, plus the effort of maintaining the model)."
        )
    cost_block = "\n".join(cost_lines) if cost_lines else "- Include the LLM in the run (`--methods llm`) to see per-ticket cost."

    skipped_block = "\n".join(f"- **{m}**: {why}" for m, why in skipped.items()) or "- none"

    unparsed = {m: r["unparsed"] for m, r in results.items() if r["unparsed"]}
    unparsed_note = (
        "\n- Some LLM replies could not be mapped to a label and were counted as wrong: "
        + ", ".join(f"{m}: {u}" for m, u in unparsed.items())
        if unparsed else ""
    )

    if unparsed_note:
        notes = unparsed_note
    elif llm_name:
        notes = "- All LLM replies were parsed successfully."
    else:
        notes = "- (no notes)"

    text = f"""# Fine-tuning vs. Prompting: Results

Test set: {n} tickets ({count_text}). The test set deliberately contains more indirect and
*novel* wording than the training data. Novel means phrasing that never appears in training.

{header}
{divider}
{chr(10).join(rows)}

## Headlines

{chr(10).join(findings)}

## Cost and speed

{cost_block}

## How to read this

- **Direct** tickets contain obvious keywords, so almost any method does well.
- **Indirect** tickets describe the problem without keywords, using phrasing the training data covers.
- **Novel** tickets use phrasing the training data never showed. This group shows how well each
  approach *generalizes*, which is what production traffic looks like.
- Latency is measured one ticket at a time, on your machine. The LLM number includes the network round trip.
- The data is synthetic, so treat differences of a few points as noise and re-run on your own tickets before deciding.

## Notes
{notes}

**Not run:**
{skipped_block}

Charts: `figures/accuracy_by_wording.png` and one confusion matrix per method in `figures/`.
"""
    Path(path).write_text(text)
