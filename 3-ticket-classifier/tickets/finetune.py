"""Approach A: fine-tune a small pretrained transformer (DistilBERT) on our labelled tickets.

We use a plain PyTorch training loop on purpose: every step is visible, and it
does not break when the `Trainer` API changes between library versions.

Run:  python -m tickets.finetune
Needs: pip install torch transformers   (first run downloads ~250 MB of weights)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from tickets.common import Prediction
from tickets.config import (
    BASE_MODEL,
    BATCH_SIZE,
    EPOCHS,
    FINETUNED_DIR,
    LABELS,
    LEARNING_RATE,
    MAX_LENGTH,
    RANDOM_STATE,
    WARMUP_RATIO,
)
from tickets.data import load_datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TicketDataset(Dataset):
    """Tokenizes up front; padding happens per batch (faster than padding everything to max length)."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.encodings = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.labels = [int(x) for x in labels]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


@torch.no_grad()
def _evaluate(model, loader, device) -> tuple[list[int], list[int]]:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
        y_pred += logits.argmax(dim=-1).cpu().tolist()
        y_true += batch["labels"].cpu().tolist()
    return y_true, y_pred


def train(
    base_model: str = BASE_MODEL,
    output_dir: Path = FINETUNED_DIR,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    max_length: int = MAX_LENGTH,
    seed: int = RANDOM_STATE,
) -> dict:
    set_seed(seed)
    device = pick_device()
    print(f"Device: {device}")

    data = load_datasets()
    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(LABELS), id2label=id2label, label2id=label2id
    ).to(device)

    collate = DataCollatorWithPadding(tokenizer)
    make_ds = lambda df: TicketDataset(df["text"], df["label"].map(label2id), tokenizer, max_length)
    train_loader = DataLoader(make_ds(data["train"]), batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(make_ds(data["val"]), batch_size=64, collate_fn=collate)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += loss.item()

        y_true, y_pred = _evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": running_loss / len(train_loader),
                        "val_accuracy": val_acc, "val_macro_f1": val_f1})
        print(f"epoch {epoch}/{epochs} | loss {history[-1]['train_loss']:.4f} | "
              f"val acc {val_acc:.3f} | val macro-F1 {val_f1:.3f}")

        if val_f1 > best_f1:  # keep the best epoch, judged on VALIDATION data (never the test set)
            best_f1 = val_f1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    summary = {
        "base_model": base_model,
        "device": str(device),
        "epochs": epochs,
        "train_seconds": time.perf_counter() - start,
        "best_val_macro_f1": best_f1,
        "n_train": len(data["train"]),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved best model to {output_dir} (train time {summary['train_seconds']:.0f}s)")
    return summary


class FineTunedClassifier:
    """Loads the saved model and classifies tickets one at a time (like an API request)."""

    def __init__(self, model_dir: Path = FINETUNED_DIR, max_length: int = MAX_LENGTH):
        if not (Path(model_dir) / "config.json").exists():
            raise FileNotFoundError(f"No fine-tuned model in {model_dir}. Run `python -m tickets.finetune` first.")
        self.device = pick_device()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[Prediction]:
        predictions = []
        for text in texts:
            start = time.perf_counter()
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            label_id = int(self.model(**enc).logits.argmax(dim=-1).item())  # .item() syncs the GPU
            predictions.append(Prediction(label=self.id2label[label_id], latency_s=time.perf_counter() - start))
        return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT on the ticket data.")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--output", default=FINETUNED_DIR)
    args = parser.parse_args()
    train(args.base_model, Path(args.output), args.epochs, args.batch_size, args.lr, args.max_length)


if __name__ == "__main__":
    main()
