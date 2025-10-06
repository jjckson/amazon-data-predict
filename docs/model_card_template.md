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

## AI 增强能力
- **摘要/聚类上线流程：** 记录需求立项、离线评估、灰度验证、正式发布的审批里程碑，明确涉及的数据资产和模型版本。
- **人工审核步骤：** 描述标注或运营团队在摘要生成、主题聚类结果发布前的抽检比例、审核准则、双人复核与升级路径。
- **失败兜底策略：** 定义模型置信度阈值、触发退回到规则或人工处理的条件、以及面向用户或下游系统的降级通知流程。

## Change History
| Release | Date | Summary | Owner |
| --- | --- | --- | --- |
| vX.Y.Z | YYYY-MM-DD | Short description of changes and rationale | Name |

> **Reminder:** Append a new row for every release and link to supporting analysis or dashboards
> when available.
