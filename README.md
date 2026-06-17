# Property Maintenance Priority Model

Deployable reference implementation for maintenance priority scoring in property management.

This repo trains, packages, and scores a transparent logistic-regression model using public NYC 311/HPD housing service request data. It estimates whether a maintenance-style service request is likely to need priority follow-up based on complaint type, borough, seasonality, channel, and location features. The workflow is structured like an internal analytics component: source-attributed data pulls, feature engineering, repeatable training, artifact export, CLI inference, tests, and containerized execution.

## System Capabilities

- downloads and prepares public NYC 311/HPD housing service request data
- converts service request fields into a numeric ML feature matrix
- trains a transparent urgent-follow-up classifier without AutoML
- exports a versioned JSON artifact with model parameters and metrics
- scores new service requests through a reusable Python inference module
- keeps outputs explainable for maintenance and operations review
- includes unit tests and CI configuration for repeatable validation

## Data Boundary

This repo does not include internal work orders, resident information, staff notes, photos, or company maintenance data. It pulls public NYC Open Data and generates local raw/processed files under `data/`. To adapt this system internally, replace the public service-request dataset with an approved work-order extract using the documented schema.

## Public Data Source

- NYC Open Data, 311 Service Requests from 2010 to Present, dataset `erm2-nwe9`

See `DATA_SOURCES.md` for endpoint details and attribution.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/train_priority_model.py
python scripts/predict.py
python -m unittest discover -s tests
```

## Dashboard

Start the maintenance-priority analytics dashboard from the project root:

```bash
streamlit run app/streamlit_app.py
```

The dashboard loads `data/processed/nyc_311_hpd_priority_training.csv` when available, scores rows with `artifacts/maintenance_priority_model.json`, and lets you upload another CSV from the sidebar. If richer internal columns such as property, building, or unit are present, the operations views use them automatically. If those columns are missing, the dashboard falls back to available public-data fields such as borough, ZIP, issue category, status, and closure-days aging.

## Notebook Walkthrough

Open `notebooks/01_nyc_311_priority_modeling.ipynb` in VS Code or Google Colab to review the full analysis flow: public 311 data pull, cleaning, target construction, feature engineering, summary measures, correlations, train/test split, logistic regression, metrics, visualizations, and model interpretation.

The training script writes:

```text
artifacts/maintenance_priority_model.json
```

The model trains on public service-request data. Metrics are useful for reproducibility and method checking, not for direct operational escalation without local validation.

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

This system is not a replacement for maintenance judgment. Any operational deployment should add data governance, threshold calibration, monitoring, access controls, approved internal work-order data, and documented human review procedures.
