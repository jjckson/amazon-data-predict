# Amazon ASIN 数据流水线（中文使用手册）

本仓库面向初学者提供了一套可落地的 Amazon ASIN 数据分析与建模方案，涵盖数据采集、清洗、特征构建、评分、训练、推理、报表以及运维治理。本文档以“分步导航”的方式，帮助你从零搭建并运行整条流水线。

## 📌 导航速览

- [新手快速体验路线图](#新手快速体验路线图)
- [环境准备与项目结构](#环境准备与项目结构)
- [端到端数据流水线手册](#端到端数据流水线手册)
  - [1. 凭证与配置准备](#1-凭证与配置准备)
  - [2. 采集与入湖](#2-采集与入湖)
  - [3. 标准化与质量校验](#3-标准化与质量校验)
  - [4. 特征构建](#4-特征构建)
  - [5. 爆发度评分](#5-爆发度评分)
  - [6. 数据导出与报表](#6-数据导出与报表)
- [模型训练与评估手册](#模型训练与评估手册)
- [在线推理与服务手册](#在线推理与服务手册)
- [报表/BI 与 AI 扩展功能](#报表bi-与-ai-扩展功能)
- [监控、治理与排障指南](#监控治理与排障指南)
- [Docker 部署全集](#docker-部署全集)
- [常见问题与后续规划](#常见问题与后续规划)

> **阅读建议**：按顺序完成“新手快速体验路线图”的三步操作后，可根据需求跳转到对应章节深入学习；所有章节标题均可在目录中直接点击跳转。

## 新手快速体验路线图

1. **克隆项目并安装依赖**
# Amazon ASIN 数据流水线

本仓库提供了一套端到端的日常 Amazon ASIN 数据管线，覆盖数据采集、标准化、特征构建与基础爆发度评分，支持后续的商业智能分析与机器学习训练。

## 项目结构总览

```
config/                 # 环境配置与凭证模板
connectors/             # 带限流与重试逻辑的 API 客户端
pipelines/              # 采集、ETL、特征与评分的编排脚本
storage/                # 数据库 DDL、特征仓库 schema、迁移脚本
utils/                  # 日志、配置、校验等通用工具
jobs/                   # 调度资产（cron 与 Airflow DAG）
tests/                  # connectors 与 pipelines 的单元测试
```

## 快速开始

1. **Install dependencies** (requires Python 3.11 or newer)
   ```bash
   git clone https://github.com/your-org/amazon-data-predict.git
   cd amazon-data-predict
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **填好凭证并运行一次干净的采集 → 清洗 → 评分流程**
2. **配置环境变量与参数**
   - 复制 `config/secrets.example.env` 为 `.env`，填充实际的令牌与数据库密码。
   - 根据部署环境调整 `config/settings.yaml`，包括限流策略、时间窗口与存储目标。
3. **运行测试**
   ```bash
   cp config/secrets.example.env .env   # 按提示填写 Amazon/数据库凭证
   python -m pipelines.ingest_raw --date 2024-05-01
   python -m pipelines.etl_standardize --date 2024-05-01
   python -m pipelines.score_baseline --date 2024-05-01
   ```
3. **打开周报或仪表盘示例验证结果**
   ```bash
   python -m reporting.weekly_report --date 2024-05-01 --output reports/
   ```
   如果需要可视化，可连接至数据库或查看 `reports/weekly_report_2024-05-01.xlsx`。

完成以上步骤后，你已完成从数据采集到结果消费的最小闭环，接下来可按导航自由探索。

## 环境准备与项目结构

### 必备工具

- Python 3.10+（建议使用虚拟环境）
- PostgreSQL 或兼容数据库（用于存储数据与特征）
- 可选：Docker / Docker Compose（用于一键部署）
- 可选：MLflow（跟踪模型训练）

### 项目结构一览

```
config/                 # 环境配置与凭证模板
connectors/             # 带限流与重试逻辑的 API 客户端
pipelines/              # 采集、ETL、特征与评分的编排脚本
storage/                # 数据库 DDL、特征仓库 schema、迁移脚本
reporting/              # SQL 查询与周报、仪表盘生成脚本
training/               # 模型训练、注册与评估工具
inference/              # 在线推理脚本与服务配置
ops/                    # Docker、调度、运维相关资源
utils/                  # 日志、配置、校验等通用工具
tests/                  # connectors、pipelines 等单元测试
```

建议先浏览 `config/`、`pipelines/` 与 `reporting/` 目录，了解参数配置与数据流向。

## 端到端数据流水线手册

### 1. 凭证与配置准备

1. 复制 `.env` 模板并填写必需变量：
   ```bash
   cp config/secrets.example.env .env
   ```
   - `AMAZON_PARTNER_TAG`、`AMAZON_ACCESS_KEY`、`AMAZON_SECRET_KEY`：Amazon API 凭证。
   - `DATABASE_URL`：PostgreSQL 连接串（示例：`postgresql://user:pass@localhost:5432/asin`）。
2. 核对 `config/settings.yaml`：
   - `ingest.rate_limit` 控制 API 调用频率。
   - `storage.raw_bucket`/`warehouse.schema` 指定数据落地位置。
   - `ai.enabled` 默认为 `false`，如需启用 AI 摘要、关键词功能请在上线前人工置为 `true` 并配置对应表名/路径。
3. 根据环境导入数据库 schema：
   ```bash
   psql $DATABASE_URL -f storage/ddl.sql
   ```
## 数据管线组件

| 脚本 | 作用 | 关键职责 |
| --- | --- | --- |
| `pipelines/ingest_raw.py` | 采集 ASIN 原始数据 | 调用产品主体、时间序列、评论、关键词等接口，持久化 JSON 并校验计数 |
| `pipelines/etl_standardize.py` | 标准化每日事实表 | 去重、异常处理、强制执行 schema，生成质量报告 |
| `pipelines/features_build.py` | 构建滚动特征 | 计算 BSR 趋势、价格波动、评论速度、Listing 质量 |
| `pipelines/score_baseline.py` | 计算基础爆发度评分 | 按站点/品类进行稳健 Z-Score，并支持权重配置 |

所有脚本均暴露 `run` 方法供编排器复用，并提供命令行入口（参见 `jobs/cron.md`）。

## 存储层

- `storage/ddl.sql`：主 PostgreSQL schema，包含维度、原始事实、数据集市、特征、评分表。
- `storage/feature_store_schema.sql`：供仪表盘与模型训练使用的视图定义。
- `storage/migrations/`：未来 Flyway/Alembic 迁移的预留目录。

## 调度策略

- `jobs/cron.md` 记录了生产环境的 UTC 每日运行节奏。
- `jobs/airflow_dags.py` 提供顺序执行与重试控制的 Airflow DAG 示例。

## 监控与数据质量

- `utils/logging.py` 支持集中式 JSON 日志输出，可接入 ELK 或 Datadog。
- `utils/rate_limiter.py`、`utils/backoff.py` 确保 connectors 遵守限流并处理 429/5xx。
- `utils/validators.py` 用于每日覆盖率/异常检测，可挂接到告警通道。

## AI Components & Governance

- **Enabling AI features:**
  1. Populate `config/ai_settings.yaml` (or create from `config/ai_settings.example.yaml`) with provider keys, summarisation limits, and safety filters.
  2. Enable the summarisation/cluster jobs in the orchestrator by toggling the `enable_ai_features` flag in `pipelines/settings.py` and redeploying the scheduler (cron/Airflow).
  3. Provision observability dashboards that track model latency, confidence scores, and rejection counts prior to broad release.
- **Human review checklist:** Ensure reviewers have access to the AI output queue, apply the documented acceptance rubric (coverage, factuality, tone), record approvals in the governance tracker, and escalate unresolved items to the domain lead within 24 hours.
- **Safety & compliance:** Confirm content filters align with policy requirements, store AI prompts/responses per retention rules, redact personal data before processing, and document periodic audits in the model registry governance log.

### 2. 采集与入湖

- 核心脚本：`pipelines/ingest_raw.py`
- 用途：调用产品主体、时间序列、评论、关键词等 API，将原始 JSON 写入对象存储或数据库原始表。
- 快速运行：
  ```bash
  python -m pipelines.ingest_raw --date 2024-05-01 --marketplace US --limit 500
  ```
- 常见参数：
  - `--date`：对齐业务日；缺省时默认今日。
  - `--marketplace`：站点代码（US、JP、DE 等）。
  - `--limit`：控制单次采集 ASIN 数量，推荐小白从 100-500 开始。
- 运行完成后，可在 `storage/raw/`（或数据库 `raw_` 表）中查看原始数据，同时 `logs/` 中会生成采集报告。

### 3. 标准化与质量校验

- 核心脚本：`pipelines/etl_standardize.py`
- 用途：结构化原始 JSON、统一数据类型、剔除异常记录，并生成质量指标。
- 快速运行：
  ```bash
  python -m pipelines.etl_standardize --date 2024-05-01
  ```
- 验收方法：
  - 检查数据库中 `fact_product_daily` 等事实表是否出现对应日期数据。
  - 阅读 `output/quality_report_2024-05-01.json` 了解缺失率、重复率等关键指标。

### 4. 特征构建

- 核心脚本：`pipelines/features_build.py`
- 用途：依据标准化事实表计算滚动窗口指标，包括 BSR 趋势、价格波动、评论速度、Listing 质量等。
- 快速运行：
  ```bash
  python -m pipelines.features_build --date 2024-05-01
  ```
- 结果位置：
  - 数据库存储于 `feature_product_window` 等表。
  - `output/features_summary_2024-05-01.csv` 提供人工审阅用的特征汇总。

### 5. 爆发度评分

- 核心脚本：`pipelines/score_baseline.py`
- 用途：按站点/品类执行稳健 Z-Score，得到综合爆发度评分，可调整权重。
- 快速运行：
  ```bash
  python -m pipelines.score_baseline --date 2024-05-01 --weights config/score_weights.example.yaml
  ```
- 输出检查：
  - 数据库表 `score_baseline_daily` 存储最终结果。
  - `output/score_rank_2024-05-01.parquet` 便于 BI 或 Notebook 继续分析。

### 6. 数据导出与报表

- 可执行 SQL：`reporting/dashboard_queries.sql` 中包含用于仪表盘的视图/查询；若启用 AI 功能，会 LEFT JOIN `ai_comment_summaries` 与 `ai_keyword_clusters`。
- 周报脚本：
  ```bash
  python -m reporting.weekly_report --date 2024-05-01 --output reports/
  ```
  - 默认会生成 Excel/Markdown 两种格式。
  - 若 AI 功能关闭，将自动回退到传统周报内容；若开启但查询失败，也会自动降级并在文档中显示“待运营审核”占位内容。
- BI 配置提醒：`reporting/dashboard_queries.sql` 中注明 `-- TODO(manual): BI 图表配置由运营确认`，上线前请由业务运营最终确认各图表映射关系。

## 模型训练与评估手册

1. **准备训练数据**：确保上述流水线已生成所需特征与标签（通常在 `feature_store` 视图中整合）。
2. **启动训练**：
   ```bash
   python -m training.train_rank \
     --train-start 2024-04-01 --train-end 2024-04-30 \
     --valid-start 2024-05-01 --valid-end 2024-05-07 \
     --output-dir runs/rank_v1
   ```
3. **查看结果**：
   - `runs/rank_v1/metrics.json`：整体指标与曲线原始数据。
   - `runs/rank_v1/metrics.csv`：适合导入表格工具快速比较。
   - `runs/rank_v1/lift_curve.csv`：查看不同阈值下的召回率和提升幅度。
4. **注册模型**（可选）：
   ```bash
   python -m training.registry register \
     --name rank_v1 \
     --artifact-uri runs:/rank_v1/model \
     --signature runs/rank_v1/signature.json \
     --metric ndcg_10=0.42
   ```
   若未部署 MLflow，可添加 `--local --registry-file runs/registry.json`，以 JSON 方式记录版本。

## 在线推理与服务手册

1. **离线批量推理**：
   ```bash
   python -m inference.batch_predict \
     --score-date 2024-05-01 \
     --model-uri runs:/rank_v1/model \
     --output-path exports/batch_predictions_2024-05-01.parquet
   ```
2. **在线服务骨架**：`inference/service/` 提供 FastAPI 示例，可通过 `uvicorn inference.service.app:app --reload` 启动开发服务器。
3. **模型热更新建议**：结合 `training/registry.py` 的 `promote` 命令验证签名兼容后再切换生产模型。

## 报表/BI 与 AI 扩展功能

- **仪表盘数据**：`reporting/dashboard_queries.sql` 提供视图定义，可直接被 BI 工具（如 Tableau、Power BI）消费。
- **AI 摘要/关键词**：
  - 启用方式：在 `config/settings.yaml` 中将 `ai.enabled` 设为 `true` 并配置 `ai.comment_summary_table`、`ai.keyword_cluster_table`。
  - 周报扩展：`reporting/weekly_report.py` 会自动新增“AI 评论摘要”与“AI 关键词聚类”章节，默认文案为“待运营审核”，运营确认后可替换为正式内容。
- **干跑工具**：`reporting/dry_run_dashboard_queries.py` 支持在无真实 AI 表时模拟数据，帮助运营提前检查图表结构。

## 监控、治理与排障指南

- **日志体系**：`utils/logging.py` 输出 JSON 结构，便于接入 ELK/Datadog；建议在生产环境设置统一的日志采集器。
- **限流与重试**：`utils/rate_limiter.py`、`utils/backoff.py` 内置指数退避策略，应对 429/5xx 响应。
- **质量验证**：`utils/validators.py` 支持缺失率、异常值检测，并可将结果推送至告警渠道。
- **故障排查速查表**：
  | 症状 | 建议排查 |
  | --- | --- |
  | 频繁出现 429 响应 | 降低 `config/settings.yaml` 中的 QPM 或联系 Amazon 支持调整限流 |
  | 缺失每日数据 | 核对质量报告与原始载荷计数，必要时重新拉取当日数据 |
  | 评分波动较大 | 查看品类 MAD/权重配置，必要时重新校准特征标准化策略 |

## Docker 部署全集

使用 Docker 可在任何环境保持一致体验，适合团队协作或快速落地。

### 1. 准备配置

```bash
cd ops/docker
cp mlflow-minio.env.example mlflow-minio.env
cp mlflow-postgres.env.example mlflow-postgres.env
# 修改凭证、端口、S3 路径等信息
```

### 2. 训练栈（含依赖）

```bash
docker compose -f docker-compose.training.yaml up --build
```

- 包含训练镜像、PostgreSQL、MinIO、MLflow。
- 进入容器：`docker compose exec trainer bash`，即可运行 `python -m training.train_rank ...`。

### 3. 推理栈

```bash
docker compose -f docker-compose.inference.yaml up --build
```

- 自动加载最新模型权重。
- 暴露 FastAPI 服务（默认 8080 端口）。

### 4. 独立 MLflow 服务（可选）

```bash
docker compose -f docker-compose.mlflow.yaml up --build
```

- Traefik 负责反向代理与 TLS。
- 复制 `traefik/users.htpasswd.example` 为 `users.htpasswd` 后填写真实账号。

> **提示**：如需从宿主机访问 MinIO 或 PostgreSQL，请确保 `.env` 与 Compose 文件中的端口映射一致，并在防火墙中放行相关端口。

## 常见问题与后续规划

- **我可以只运行部分流程吗？** 可以。跳过某一阶段前，请确保下游脚本有对应的输入数据或使用干跑数据。
- **如何编排定时任务？** 参考 `jobs/cron.md` 或 `jobs/airflow_dags.py`，里面提供了 cron 表达式与 Airflow DAG 示例。
- **如何维护模型卡？** 使用 [`docs/model_card_template.md`](docs/model_card_template.md)，发布时复制为新版本并更新指标、审批历史。
- **后续规划**：
  - 扩展 BI 仪表盘与 AI 评分的联动策略。
  - 强化数据验证与自动化回滚机制。
  - 丰富持续交付与灰度发布工具链。

祝你使用顺利！如遇问题，先参考本 README 对应章节，再查看 `tests/` 中的示例或提交 Issue 与我们交流。
评分管线会将结果写入 `score_baseline_daily`，可配合 `storage/feature_store_schema.sql` 中的视图供 BI 仪表盘或 CSV 导出（如 `exports/top_candidates_YYYYMMDD.csv`）。

## 本地开发建议

- 迭代开发时，可使用 `pipelines/ingest_raw.py` 搭配 ≤500 个 ASIN，避免触发限流。
- 在 `tests/` 中的夹具基础上模拟 connectors，参考现有单元测试写法。
- 所有凭证仅存放于 `.env`，严禁提交到版本库。

## 排名模型训练

`training/train_rank.py` 支持训练 LambdaRank 模型，并输出 group 维度的 `ndcg@k`、`recall@k`、MAP 与相对基线的提升幅度。如需对比历史基线，可通过 `--baseline-score-column` 提供参考列。脚本会在 `--output-dir` 下生成：

- `metrics.json`：聚合指标与提升曲线原始数据。
- `metrics.csv`：便于表格工具读取的指标快照。
- `lift_curve.csv`：记录各切点的召回率、基线召回率与 Lift。

所有指标会通过 `RunLogger` 记录，方便追踪实验。

## A/B 测试工具

`abtest/` 目录下提供了两类辅助模块：

- `abtest/traffic_split.py`：`TrafficSplitter` 基于哈希将 ASIN（或任意 ID）划分实验桶，可按全局或品类自定义流量，并记录可选的日志轨迹。
- `abtest/uplift_eval.py`：`evaluate_uplift` / `UpliftEvaluator` 用于汇总离线曝光日志，计算 CTR、CR、GMV、ROAS 等指标相对对照组的提升，并提供正态近似的显著性检验。

### 离线评估流程

1. 构建包含 `variant`、`impressions`、`clicks`、`conversions`、`gmv`、`spend`（可选 `gmv_sq`、`roas`、`roas_sq`）的聚合数据框。
2. 调用 `evaluate_uplift(df, control_variant="control")` 或在脚本/Notebook 中实例化 `UpliftEvaluator`。
3. 使用 `report.to_dataframe()` 导出结构化结果，或 `report.to_markdown()` 快速生成 Markdown 报告。

参见 `tests/test_abtest.py` 获取端到端示例。

## 故障排查

| 症状 | 建议排查 | 
| --- | --- |
| 频繁出现 429 响应 | 降低 `config/settings.yaml` 中的 QPM，或检查租户级限流设置 |
| 缺失每日数据 | 检查校验报告与原始载荷计数 |
| 评分波动较大 | 审核站点/品类的 MAD，必要时调整配置权重 |

## 模型卡治理

- 模板位于 [`docs/model_card_template.md`](docs/model_card_template.md)，每次发布复制为 `docs/model_card_vX.Y.Z.md` 并补充最新数据、指标与审批。
- 在已发布的模型卡 **Change History** 中新增行，记录发布标签、上线日期、行为变化摘要与负责人。
- 根据验证材料更新 **Data Scope**、**Metrics**、**Known Limits**、**Compliance & Governance** 等章节。
- 合并发布分支后，将模型卡链接同步到分析注册表或部署追踪系统。

## 模型注册运维

`training/registry.py` 提供本地 JSON 与 MLflow Registry 双模式的模型版本管理 CLI：

```bash
# 在当前 MLflow Tracking URI 下注册模型版本
python -m training.registry register \
  --name rank_v1 \
  --artifact-uri runs:/abc123/model \
  --signature signature.json \
  --metric auc=0.91 \
  --data-span 2024-01-01:2024-01-15 \
  --feature-version features:v5

# 与线上部署比对签名后提升到生产
python -m training.registry promote --name rank_v1 --version 2 --to prod

# 查看模型版本列表
python -m training.registry list --name rank_v1
```

在缺少 MLflow（或未安装 `mlflow`）的情况下，可使用 `--local --registry-file runs/registry.json` 保存在地 JSON。提升到 *Staging* 或 *Production* 时会校验签名与特征契约，确保兼容性。JSON Registry 会记录 `artifact_uri`、`metrics`、`data_span`、`feature_version`、`signature`，便于离线环境复现。

## Docker 部署指引

为了简化环境一致性，可使用 `ops/docker/` 下的 Compose 文件完成训练、推理与 MLflow 服务编排。

1. **准备配置文件**
   ```bash
   cd ops/docker
   cp mlflow-minio.env.example mlflow-minio.env
   cp mlflow-postgres.env.example mlflow-postgres.env
   # 根据需求更新凭证、端口等变量
   ```
2. **启动训练栈**
   ```bash
   docker compose -f docker-compose.training.yaml up --build
   ```
   该栈会构建 `ops/docker/Dockerfile.training`，并联动 PostgreSQL、MinIO、MLflow 等依赖，便于在容器内触发 `training/train_rank.py`。
3. **启动推理栈**
   ```bash
   docker compose -f docker-compose.inference.yaml up --build
   ```
   复用相同的依赖，加载最新模型，运行基于 `ops/docker/Dockerfile.inference` 构建的推理服务。
4. **仅启动 MLflow 服务**（可选）
   ```bash
   docker compose -f docker-compose.mlflow.yaml up --build
   ```
   该栈通过 Traefik 反向代理暴露 MLflow UI。启动前，请在 `traefik/` 目录内放置 TLS 证书，并将 `users.htpasswd.example` 复制为 `users.htpasswd` 后填入真实凭证。

> **提示**：若需在容器外访问 MinIO 或 PostgreSQL，请确保 `.env` 与 Compose 文件中的端口映射保持一致，并在本地网络策略中开放相应端口。

## 后续规划

- 扩展 BI 仪表盘与 AI 评分的集成策略。
- 强化数据验证与自动化回滚机制。
- 丰富持续交付与灰度发布工具链。
