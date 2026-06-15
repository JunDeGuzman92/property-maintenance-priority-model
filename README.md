# Property Maintenance Priority Model

Deployable reference implementation for maintenance priority scoring in property management.

This repo trains, packages, and scores a transparent logistic-regression model that estimates whether a work order should be escalated for urgent follow-up. It uses synthetic work-order data, but the workflow is structured like an internal analytics component: feature engineering, repeatable training, artifact export, CLI inference, tests, and containerized execution.

## System Capabilities

- converts operational work-order fields into a numeric ML feature matrix
- trains a transparent urgent-follow-up classifier without AutoML
- exports a versioned JSON artifact with model parameters and metrics
- scores new work orders through a reusable Python inference module
- keeps outputs explainable for maintenance and operations review
- includes unit tests and CI configuration for repeatable validation

## Data Boundary

All data is synthetic. No real work orders, resident information, staff notes, photos, or internal maintenance data are included. To adapt this system internally, replace the sample CSV with an approved governed extract using the documented schema.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train_priority_model.py
python scripts/predict.py
python -m unittest discover -s tests
```

The training script writes:

```text
artifacts/maintenance_priority_model.json
```

Because the dataset is intentionally small and synthetic, training metrics are a reproducibility check, not a production-performance claim.

## CLI Scoring

Score the built-in sample record:

```bash
python scripts/predict.py
```

Score a specific work order:

```bash
python scripts/predict.py --input-json "{\"age_days\": 4, \"category\": \"hvac\", \"occupied_unit\": 1, \"safety_flag\": 1, \"recurrence_count\": 2, \"asset_age_years\": 14, \"after_hours\": 1}"
```

## Container Run

```bash
docker build -t property-maintenance-priority-model .
docker run --rm property-maintenance-priority-model
```

## Responsible Use

This system is not a replacement for maintenance judgment. Any operational deployment should add data governance, threshold calibration, monitoring, access controls, and documented human review procedures.
