# Model Card: Maintenance Priority Classifier

## Model Summary

This is a small logistic-regression classifier trained on public NYC 311/HPD housing service request data. It estimates whether a maintenance-style service request is likely to need priority follow-up and exports a portable JSON artifact for reproducible inference.

## Intended Use

This model is intended as a deployable reference implementation for maintenance triage workflows. It can support:

- feature engineering for operational service-request data
- model training and testing
- repeatable inference from a saved artifact
- human-in-the-loop maintenance prioritization
- controlled internal prototyping against approved public or internal data

It should not be used for operational escalation decisions without validation on approved data, threshold calibration, and documented human oversight.

## Training Data

The training data is generated from public NYC Open Data:

- NYC Open Data, 311 Service Requests from 2010 to Present, dataset `erm2-nwe9`

No real work orders, resident details, internal staff notes, photos, or maintenance records are included.

## Features

- complaint type
- borough
- service request month
- winter flag
- location type
- public reporting channel
- descriptor availability

## Limitations

- Public 311/HPD requests are not the same as internal work orders.
- Closure time can be affected by agency workflow, reporting delay, and data quality.
- The model is intentionally simple and transparent.
- Any real deployment would require data-quality checks, privacy review, model monitoring, and human oversight.
