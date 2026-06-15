# Data Sources

This repo uses public datasets only. The training script does not require company data.

## NYC Open Data: 311 Service Requests from 2010 to Present

- Provider: City of New York
- Dataset: 311 Service Requests from 2010 to Present
- Dataset ID: `erm2-nwe9`
- API endpoint: `https://data.cityofnewyork.us/resource/erm2-nwe9.json`
- Use in this repo: HPD housing complaint records are used as a public proxy for maintenance-style service requests.

## Important Boundary

NYC 311/HPD data is not the same as an internal maintenance work-order system. It is useful for public, reproducible modeling practice around complaint categories, closure times, and follow-up risk. Any internal workflow should replace it with approved work-order data and documented escalation rules.
