"""Synthetic customer-support tickets, split into train / validation / test.

WHY SYNTHETIC: it lets the project run offline and reproducibly. It is only a
stand-in. Real tickets are messier, so swap in your own CSV (columns: text,label)
when you can.

WHAT MAKES THIS DATASET USEFUL FOR THE COMPARISON
Every ticket has a `wording` tag:
  direct   - uses obvious keywords ("refund", "password", "crashes")
  indirect - describes the problem without the obvious keywords
  novel    - indirect phrasing that NEVER appears in the training data
The test set deliberately contains more indirect and novel tickets than the
training set. Keyword-based models look great on `direct` tickets, so the
interesting question is how each approach does on the harder two groups.

Usage:
    python -m tickets.data
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from tickets.config import (
    DATA_DIR,
    LABELS,
    N_TEST,
    N_TRAIN_VAL,
    RANDOM_STATE,
    VAL_FRACTION,
)

# Each label has three phrase banks: direct, indirect (seen in training), novel (test only).
BANKS = {
    "billing": {
        "direct": [
            "I was charged twice for {product} this month.",
            "There is an incorrect charge on my latest invoice.",
            "Why did my subscription price go up? My bill is higher than usual.",
            "I need to update the credit card on file for my billing.",
            "Can you send me a copy of my invoice for last month?",
            "My payment failed but the money was still taken from my card.",
            "I was billed after I cancelled my subscription.",
            "The amount on my invoice does not match the plan I signed up for.",
        ],
        "indirect": [
            "My bank statement shows two identical transactions from you this month.",
            "The total I paid last week is way more than the price listed on your website.",
            "I keep getting emails saying my card was declined, but it works everywhere else.",
            "Something looks off with the numbers on my monthly statement.",
        ],
        "novel": [
            "Looking at my credit card activity, your company has taken more than I agreed to pay.",
            "Accounting needs a proper receipt for what we paid you, and the one I got shows the wrong amount.",
        ],
    },
    "refund_return": {
        "direct": [
            "I would like a refund for {product}.",
            "How do I return {product}? It does not fit.",
            "I want my money back, {product} is not what I expected.",
            "Please process a return for my order.",
            "The item arrived damaged and I want to send it back for a full refund.",
            "I changed my mind and would like to return {product} within the return window.",
            "Can I get a refund for the unused months of my plan?",
            "I need a return label for {product}.",
        ],
        "indirect": [
            "This is not what I expected, I would like to send it back and get my money returned.",
            "I do not want this anymore, what are my options to be reimbursed?",
            "It broke after two days, I do not want a replacement, I just want to be paid back.",
            "Please take this back, it does not do what the description promised.",
        ],
        "novel": [
            "We ended up not needing {product}, so how can we ship it back and recover the payment?",
            "The quality is far below what was advertised, so I am giving it back and expect my funds returned.",
        ],
    },
    "shipping": {
        "direct": [
            "My package has not arrived yet and it was supposed to be here days ago.",
            "Where is my order? The tracking page has not updated in a week.",
            "The delivery address on my order is wrong, can you change it?",
            "My order shows as delivered but I never received it.",
            "How long will shipping take to my country?",
            "The courier keeps missing me, can I reschedule the delivery?",
            "Part of my order is missing from the box that was delivered.",
            "Tracking says my parcel is stuck at the local depot.",
        ],
        "indirect": [
            "It has been over two weeks and there is still nothing on my doorstep.",
            "The estimated arrival date came and went and I have heard nothing.",
            "I am travelling next week, will my parcel get here before I leave?",
            "A neighbour said a delivery person left a card but I never saw a package.",
        ],
        "novel": [
            "I ordered this as a birthday present and the party is tomorrow, but nothing has turned up at my place.",
            "Your carrier's site says 'in transit' and has said that since the first of the month.",
        ],
    },
    "technical_issue": {
        "direct": [
            "The app crashes every time I open it.",
            "I keep getting error code 500 when I try to save my work.",
            "{product} will not connect to wifi anymore.",
            "The page loads very slowly and sometimes freezes.",
            "I cannot upload files, the progress bar gets stuck at 90 percent.",
            "Notifications stopped working after the latest update.",
            "The screen goes black right after the logo appears.",
            "Sync between my phone and laptop is not working.",
        ],
        "indirect": [
            "Ever since the update, nothing responds when I tap the buttons.",
            "It worked fine yesterday but today it just shows a blank white page.",
            "Every time I try to export the report, the whole thing shuts down.",
            "The device gets really hot and turns itself off after a few minutes.",
        ],
        "novel": [
            "My colleagues can open the dashboard but on my computer it spins forever and never loads.",
            "After I installed the new version, the audio cuts out halfway through every video.",
        ],
    },
    "account_access": {
        "direct": [
            "I forgot my password and the reset email never arrives.",
            "My account is locked after too many login attempts.",
            "I cannot log in, it says my credentials are invalid.",
            "I lost my phone and cannot get the two-factor code.",
            "Please change the email address on my account.",
            "I want to delete my account and all my data.",
            "Someone else may have accessed my account, I need to secure it.",
            "I never received the verification email for my new account.",
        ],
        "indirect": [
            "I have tried every password I can think of and still cannot get in.",
            "The site keeps sending me back to the sign-in page in a loop.",
            "My old work email no longer exists, how do I move everything to my new address?",
            "I got a message saying there was a login from another country and it was not me.",
        ],
        "novel": [
            "I changed phones and now the app asks for a code that goes to a number I do not have anymore.",
            "My teammate left the company and I need to take over the account she set up.",
        ],
    },
    "feature_request": {
        "direct": [
            "It would be great if you added a dark mode.",
            "Please add an option to export my data to CSV.",
            "Can you support login with Apple in a future version?",
            "I would love to see a calendar integration.",
            "Any plans to add multiple user profiles on one account?",
            "A bulk edit feature would save me hours every week.",
            "It would be really helpful to have keyboard shortcuts.",
            "Please consider adding support for more languages.",
        ],
        "indirect": [
            "I wish the dashboard let me compare two months side by side.",
            "My whole team would switch to you if there was a way to set custom reminders.",
            "Right now I have to copy everything by hand, there should be a smarter way.",
            "Have you thought about letting us customise the home screen?",
        ],
        "novel": [
            "A competitor's tool lets you schedule reports in advance, and it would be amazing to see that here.",
            "I keep exporting to a spreadsheet just to sort things, and sorting inside the app would be so much easier.",
        ],
    },
}

PRODUCTS = ["the Pro plan", "my Nova headphones", "the mobile app", "the desktop app", "my order",
            "my subscription", "the smart lamp", "the Aero blender", "my Starter account"]
OPENERS = ["", "", "Hi,", "Hello team,", "Hey,", "Good morning,", "To whom it may concern,"]
CLOSERS = ["", "", "Thanks.", "Please help.", "Thanks in advance!", "Let me know what I can do.",
           "This is really frustrating.", "Appreciate the quick help."]
# Neutral sentences that appear for every label, so they carry no information about it.
CONTEXT = ["I have been a customer for {n} years.", "I am using the {platform} version.",
           "This is the second time I am writing about this.", "Reference: order #{order}."]


def _typo(text: str, rng: np.random.Generator) -> str:
    """Swap two adjacent letters inside one longer word, like a fast typist would."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    if not candidates:
        return text
    i = int(rng.choice(candidates))
    w = words[i]
    j = int(rng.integers(1, len(w) - 2))
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _make_ticket(label: str, wording: str, rng: np.random.Generator) -> str:
    phrase = str(rng.choice(BANKS[label][wording]))
    text = phrase.format(product=rng.choice(PRODUCTS))
    parts = [str(rng.choice(OPENERS)), text]
    if rng.random() < 0.35:
        parts.append(str(rng.choice(CONTEXT)).format(
            n=int(rng.integers(1, 9)), platform=rng.choice(["iOS", "Android", "web", "Windows"]),
            order=int(rng.integers(10000, 99999))))
    parts.append(str(rng.choice(CLOSERS)))
    text = " ".join(p for p in parts if p)
    if rng.random() < 0.15:
        text = _typo(text, rng)
    if rng.random() < 0.10:
        text = text.lower()
    return text


def _generate(n: int, wording_probs: dict[str, float], rng: np.random.Generator, seen: set) -> pd.DataFrame:
    rows = []
    wordings, probs = list(wording_probs), list(wording_probs.values())
    labels = sorted(BANKS)
    i = 0
    while len(rows) < n:
        label = labels[i % len(labels)]  # round-robin keeps classes balanced
        wording = str(rng.choice(wordings, p=probs))
        text = _make_ticket(label, wording, rng)
        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "label": label, "wording": wording})
        i += 1
    return pd.DataFrame(rows).sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)


def generate_datasets(
    n_train_val: int = N_TRAIN_VAL,
    n_test: int = N_TEST,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} DataFrames with columns text, label, wording."""
    rng = np.random.default_rng(seed)
    seen: set = set()
    pool = _generate(n_train_val, {"direct": 0.75, "indirect": 0.25}, rng, seen)
    test = _generate(n_test, {"direct": 0.50, "indirect": 0.25, "novel": 0.25}, rng, seen)
    n_val = int(round(len(pool) * val_fraction))
    return {"train": pool.iloc[n_val:].reset_index(drop=True),
            "val": pool.iloc[:n_val].reset_index(drop=True),
            "test": test}


def load_datasets(data_dir=DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load train/val/test CSVs. Custom data needs `text` and `label` (and optionally `wording`)."""
    out = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m tickets.data` first.")
        df = pd.read_csv(path)
        missing = {"text", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
        if "wording" not in df.columns:
            df["wording"] = "all"
        unknown = set(df["label"]) - set(LABELS)
        if unknown:
            raise ValueError(f"{path} has labels not listed in tickets/config.py LABELS: {sorted(unknown)}")
        out[split] = df
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic ticket dataset.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    splits = generate_datasets(seed=args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        mix = df["wording"].value_counts(normalize=True).round(2).to_dict()
        print(f"{name:<5} {len(df):>5} tickets | wording mix {mix}")
    print(f"Wrote CSVs to {DATA_DIR}")


if __name__ == "__main__":
    main()
