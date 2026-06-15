# Model Card: Maintenance Priority Classifier

## Model Summary

This is a small logistic-regression classifier trained on synthetic property-management work-order data. It estimates whether a work order should enter an urgent follow-up review queue and exports a portable JSON artifact for reproducible inference.

## Intended Use

This model is intended as a deployable reference implementation for maintenance triage workflows. It can support:

- feature engineering for operational work-order data
- model training and testing
- repeatable inference from a saved artifact
- human-in-the-loop maintenance prioritization
- controlled internal prototyping against approved data

It should not be used for operational escalation decisions without validation on approved data, threshold calibration, and documented human oversight.

## Training Data

The training data is synthetic and stored in:

`data/synthetic_work_orders.csv`

No real work orders, resident details, internal staff notes, photos, or maintenance records are included.

## Features

- work-order age
- category
- occupied-unit flag
- safety flag
- recurrence count
- asset age
- after-hours flag

## Limitations

- Synthetic data cannot represent real maintenance operations.
- The target label is simulated.
- The model is intentionally simple and transparent.
- High metrics on the toy dataset are a reproducibility signal, not a production-performance claim.
- Any real deployment would require data-quality checks, privacy review, model monitoring, and human oversight.
