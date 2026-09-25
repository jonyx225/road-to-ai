# Fine-tuning vs. Prompting: Results

Test set: 300 tickets (direct: 142, indirect: 77, novel: 81). The test set deliberately contains more indirect and
*novel* wording than the training data. Novel means phrasing that never appears in training.

| Method | Accuracy | Macro-F1 | Direct | Indirect | Novel | Median latency | p95 latency |
|---|---|---|---|---|---|---|---|
| TF-IDF + LR (baseline) | 92.7% | 0.927 | 100.0% | 100.0% | 72.8% | 0.4 ms | 0.9 ms |

## Headlines

- Highest overall accuracy: **TF-IDF + LR (baseline)** (92.7%).
- Best on the hardest group (novel wording): **TF-IDF + LR (baseline)** (72.8%).
- Fastest (median latency): **TF-IDF + LR (baseline)** (0.4 ms).

## Cost and speed

- Include the LLM in the run (`--methods llm`) to see per-ticket cost.

## How to read this

- **Direct** tickets contain obvious keywords, so almost any method does well.
- **Indirect** tickets describe the problem without keywords, using phrasing the training data covers.
- **Novel** tickets use phrasing the training data never showed. This group shows how well each
  approach *generalizes*, which is what production traffic looks like.
- Latency is measured one ticket at a time, on your machine. The LLM number includes the network round trip.
- The data is synthetic, so treat differences of a few points as noise and re-run on your own tickets before deciding.

## Notes
- (no notes)

**Not run:**
- none

Charts: `figures/accuracy_by_wording.png` and one confusion matrix per method in `figures/`.
