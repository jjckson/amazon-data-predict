# Model Card Template

Use this template to document each production release of the Explosive Score model. Update the
sections below with concise, factual information sourced from the most recent training run,
validation report, and stakeholder review.

## Data Scope
- **Training sources:** Describe the datasets, collection windows, filtering rules, and notable
  exclusions.
- **Population coverage:** Clarify marketplaces, locales, ASIN segments, and any demographic or
  product constraints.
- **Update cadence:** Note refresh frequency, retraining schedule, and triggers for ad-hoc runs.

## Metrics
- **Core evaluation metrics:** Capture offline metrics (e.g., MAP@K, precision/recall, calibration).
- **Monitoring signals:** List online KPIs or shadow deployment checks tracked post-release.
- **Benchmark comparisons:** Summarise baselines or prior models used for reference.

## Known Limits
- **Data limitations:** Enumerate sampling gaps, reporting delays, or known anomalies.
- **Model behaviour:** Highlight failure modes, bias considerations, and edge cases.
- **Operational caveats:** Document dependencies that impact availability or quality.

## Compliance & Governance
- **Approvals:** Record sign-off stakeholders and dates (product, legal, compliance, DS lead).
- **Policies:** Reference applicable governance, retention, and privacy requirements.
- **Review cadence:** State audit frequency and links to the latest review artefacts.

## Change History
| Release | Date | Summary | Owner |
| --- | --- | --- | --- |
| vX.Y.Z | YYYY-MM-DD | Short description of changes and rationale | Name |

> **Reminder:** Append a new row for every release and link to supporting analysis or dashboards
> when available.
