"""Generate a synthetic Online Retail II-like CSV.

This is a fallback for offline environments where the UCI workbook cannot be
downloaded or staged into project input. It preserves the commonly used schema:
InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID,
Country.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "temp_download" / "online_retail_ii_synthetic.csv"

PRODUCTS = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
    ("71053", "WHITE METAL LANTERN", 3.39),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 3.39),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 7.65),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
    ("22633", "HAND WARMER UNION JACK", 1.85),
    ("22632", "HAND WARMER RED POLKA DOT", 1.85),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22745", "POPPY'S PLAYHOUSE BEDROOM", 2.10),
    ("22748", "POPPY'S PLAYHOUSE KITCHEN", 2.10),
    ("22749", "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.95),
    ("22910", "PAPER CHAIN KIT VINTAGE CHRISTMAS", 2.95),
]

COUNTRIES = [
    ("United Kingdom", 0.82),
    ("Germany", 0.035),
    ("France", 0.032),
    ("EIRE", 0.025),
    ("Spain", 0.018),
    ("Netherlands", 0.012),
    ("Belgium", 0.010),
    ("Switzerland", 0.008),
    ("Portugal", 0.008),
    ("Australia", 0.007),
    ("Norway", 0.006),
    ("Italy", 0.006),
    ("Channel Islands", 0.005),
    ("Finland", 0.004),
    ("Cyprus", 0.004),
]


def weighted_country(rng: random.Random) -> str:
    roll = rng.random()
    cumulative = 0.0
    for country, weight in COUNTRIES:
        cumulative += weight
        if roll <= cumulative:
            return country
    return "United Kingdom"


def invoice_date(rng: random.Random) -> datetime:
    start = datetime(2009, 12, 1, 8, 0, 0)
    end = datetime(2011, 12, 9, 18, 0, 0)
    seconds = int((end - start).total_seconds())
    value = start + timedelta(seconds=rng.randrange(seconds))
    return value.replace(second=0)


def generate_rows(row_count: int, seed: int):
    rng = random.Random(seed)
    customer_ids = [str(customer_id) for customer_id in range(12346, 18288)]
    invoice_base = 489434
    rows_written = 0

    while rows_written < row_count:
        line_count = min(rng.randint(1, 8), row_count - rows_written)
        invoice_no = str(invoice_base)
        is_cancel = rng.random() < 0.018
        if is_cancel:
            invoice_no = "C" + invoice_no
        invoice_base += 1

        date_value = invoice_date(rng).strftime("%Y-%m-%d %H:%M:%S")
        country = weighted_country(rng)
        customer_id = "" if rng.random() < 0.23 else rng.choice(customer_ids)

        for _ in range(line_count):
            stock_code, description, base_price = rng.choice(PRODUCTS)
            quantity = max(1, int(rng.lognormvariate(1.8, 0.9)))
            if is_cancel:
                quantity = -quantity
            unit_price = max(0.08, round(base_price * rng.uniform(0.85, 1.25), 2))
            if rng.random() < 0.004:
                description = ""

            yield [
                invoice_no,
                stock_code,
                description,
                quantity,
                date_value,
                f"{unit_price:.2f}",
                customer_id,
                country,
            ]
            rows_written += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=502)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
            "Country",
        ])
        writer.writerows(generate_rows(args.rows, args.seed))

    print(f"[STATUS] Synthetic dataset saved: {output_path}")
    print(f"[STATUS] Rows: {args.rows:,}")


if __name__ == "__main__":
    main()
