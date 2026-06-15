# Property Maintenance Priority Model

Sanitized machine-learning example for maintenance triage in property management.

This repo trains a small logistic-regression model from scratch to estimate whether a work order should be escalated for urgent follow-up. It uses synthetic work-order data and is designed for reproducible internship evidence.

## What This Demonstrates

- feature engineering on operational maintenance data
- categorical encoding
- train/test evaluation
- transparent ML without external AutoML tooling
- human-in-the-loop review framing

## Privacy Boundary

All data is synthetic. No real work orders, resident information, staff notes, photos, or internal maintenance data are included.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train_priority_model.py
python -m unittest discover -s tests
```

The training script writes:

```text
artifacts/maintenance_priority_model.json
```

Because the dataset is intentionally small and synthetic, training metrics should be read only as a reproducibility check. They are not evidence of production accuracy.

## Responsible Use

This is not a replacement for maintenance judgment. It is a learning artifact showing how a review queue could prioritize work based on transparent features such as safety flag, age, recurrence, occupancy impact, and category.
