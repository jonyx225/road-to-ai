"""Central settings for the project. Change things here, not in the scripts."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "churned"          # 1 = customer left, 0 = stayed
ID_COLUMN = "customer_id"   # identifier: never a feature

# Columns that leak the answer (they only exist AFTER a customer has left).
# `python -m churn.explore` flags these; we list them here so training skips them.
LEAKY_COLUMNS = ["final_invoice_sent"]

# Set to False when you switch to real company data (removes the "synthetic" caveat from the report).
IS_SYNTHETIC_DATA = True

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


@dataclass(frozen=True)
class BusinessCosts:
    """The economics of a retention campaign. Replace with your company's numbers.

    We contact every customer the model flags as likely to leave.
      - Every contact costs `contact_cost` (discount, call, email, etc.).
      - If the customer really was going to leave, the offer works with
        probability `save_rate`, and keeping them is worth `customer_value`.
    """

    customer_value: float = 250.0
    contact_cost: float = 20.0
    save_rate: float = 0.40

    @property
    def gain_per_true_positive(self) -> float:
        """Net profit from contacting a real churner (expected saved value minus contact cost)."""
        return self.save_rate * self.customer_value - self.contact_cost

    @property
    def loss_per_false_positive(self) -> float:
        """Money wasted contacting a customer who would have stayed anyway."""
        return self.contact_cost
