# Amazon ASIN 数据管线

该仓库提供从数据采集到评分的主干流程，收集每日 Amazon ASIN 信号，标准化为关系型表结构，构建分析特征，并产出用于下游看板和模型训练的基线「爆款分」排序。

## 仓库结构

```
config/                 # 环境配置与凭证模板
connectors/             # 带限流和重试的 API 客户端
pipelines/              # 采集、ETL、特征、评分管道
storage/                # 数据库 DDL、特征仓库 schema、迁移脚本
utils/                  # 日志、配置、校验等通用工具
jobs/                   # 调度资产（cron 与 Airflow DAG）
tests/                  # connectors 与 pipelines 的单测
```

## 快速开始

1. **安装依赖**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **配置环境**
   - 复制 `config/secrets.example.env` 为 `.env` 并填写令牌/密码。
   - 根据环境、限流、时间窗口与存储目标调整 `config/settings.yaml`。
3. **运行测试**
   ```bash
   pytest
   ```

## AI 组件与治理

- **启用步骤：**
  1. 以 `config/ai_settings.example.yaml` 为模板创建 `config/ai_settings.yaml`，填写模型服务密钥、摘要长度限制和安全过滤策略。
  2. 在 `pipelines/settings.py` 中开启 `enable_ai_features` 标志，并更新调度（cron/Airflow）以加载新任务。
  3. 发布前搭建观测面板，关注延迟、置信度与拒绝次数等指标。
- **人工审核清单：** 审核人员需接入 AI 产出队列，按照覆盖度、事实准确性、语气的评分标准抽检，并在治理台账登记结果；存在争议时于 24 小时内升级至域负责人。
- **安全与合规：** 确认内容过滤符合政策要求；提示词与响应需按留存策略存档；处理前脱敏个人信息，并将周期性审核记录到模型注册治理日志中。

## 其它资源

- `docs/model_card_template.md`：用于记录每次上线的模型卡模板。
- `jobs/cron.md` 与 `jobs/airflow_dags.py`：生产调度示例。
- `storage/feature_store_schema.sql`：可用于 BI 或 CSV 导出的视图定义。

更多细节请参考英文版 README。
