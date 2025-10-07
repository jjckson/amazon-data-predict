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
   ```bash
   git clone https://github.com/your-org/amazon-data-predict.git
   cd amazon-data-predict
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **填好凭证并运行一次干净的采集 → 清洗 → 评分流程**
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
