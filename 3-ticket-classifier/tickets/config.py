"""Central settings. Change things here (or via environment variables), not in the scripts."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FINETUNED_DIR = MODELS_DIR / "distilbert-tickets"

RANDOM_STATE = 42

# ---------------------------------------------------------------- labels
LABELS = [
    "account_access",
    "billing",
    "feature_request",
    "refund_return",
    "shipping",
    "technical_issue",
]

# Definitions are given to the LLM in its prompt. Good definitions matter a lot.
LABEL_DESCRIPTIONS = {
    "account_access": "Login problems, password resets, two-factor codes, locked or hacked accounts, changing account email, deleting an account.",
    "billing": "Charges, invoices, payment methods, subscription prices, double charges, failed payments. NOT requests to give money back for a product.",
    "feature_request": "Suggestions or requests for new features or improvements to the product.",
    "refund_return": "Asking to return a product, get a refund, or be reimbursed because the product was unwanted, faulty or not as described.",
    "shipping": "Delivery status, missing or late packages, tracking, delivery address or courier problems.",
    "technical_issue": "Bugs, crashes, errors, slowness, connectivity or sync problems, or a product that stops working.",
}

# ---------------------------------------------------------------- data sizes
N_TRAIN_VAL = 1200      # tickets used for training + validation
VAL_FRACTION = 0.125    # 150 validation tickets
N_TEST = 300            # test tickets (deliberately includes harder wording)

# ---------------------------------------------------------------- fine-tuning
BASE_MODEL = "distilbert-base-uncased"
MAX_LENGTH = 128
EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1

# ---------------------------------------------------------------- LLM prompting
LLM_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
# Set LLM_TEMPERATURE=none if your chosen model rejects the temperature parameter.
_temperature = os.getenv("LLM_TEMPERATURE", "0")
LLM_TEMPERATURE = None if _temperature.lower() == "none" else float(_temperature)
LLM_MAX_TOKENS = 20
SHOTS_PER_CLASS = 3

# Price in USD per million tokens. Defaults are Claude Haiku 4.5's list prices
# (checked Sept 2026). PRICES CHANGE: verify at https://docs.claude.com and update
# these (or set the env vars) if you use a different model.
LLM_PRICE_INPUT_PER_MTOK = float(os.getenv("LLM_PRICE_INPUT_PER_MTOK", "1.00"))
LLM_PRICE_OUTPUT_PER_MTOK = float(os.getenv("LLM_PRICE_OUTPUT_PER_MTOK", "5.00"))

# ASSUMPTION for the break-even analysis: what it costs per month to keep the
# fine-tuned model running (a small always-on server). Replace with your real number.
SELF_HOST_COST_PER_MONTH = float(os.getenv("SELF_HOST_COST_PER_MONTH", "150"))
