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

1. **安装依赖**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **配置环境变量与参数**
   - 复制 `config/secrets.example.env` 为 `.env`，填充实际的令牌与数据库密码。
   - 根据部署环境调整 `config/settings.yaml`，包括限流策略、时间窗口与存储目标。
3. **运行测试**
   ```bash
   pytest
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

## 数据导出

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
