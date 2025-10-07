# 运维手册（Ops Runbook）

本手册为上线后排障与日常维护提供步骤。请根据实际环境补充具体信息。

## 1. 值班联系方式
- On-call 负责人：
- 备用联系人：
- 告警渠道：Slack #asin-alerts / 邮件 ops@example.com

## 2. 日常巡检
| 项目 | 检查频率 | 指标/脚本 | 备注 |
| ---- | -------- | ---------- | ---- |
| 数据采集成功率 | 每日 | `python -m pipelines.ingest_raw --health-check` | 关注 429 与 5xx 错误 |
| 特征落库延迟 | 每日 | `SELECT max(dt) FROM feature_product_window;` | 超过 1 天需排查 |
| 模型服务健康 | 每班次 | `curl http://inference:8080/health` | 返回 200 即正常 |
| 报表生成状态 | 每周 | `python -m reporting.weekly_report --status` | 失败需触发手动重跑 |

## 3. 常见告警处理
- **TopN 命中率过低**：
  1. 查看 `reports/rank_acceptance/metrics.csv` 与最近批次预测差异。
  2. 若模型指标显著下降，执行模型回滚（参考第 5 节）。
- **PSI 超阈值**：
  1. 运行 `python -m reporting.dry_run_alerts` 验证特征漂移样例。
  2. 检查 `storage/feature_drift/` 中的漂移报告，评估是否需重新训练。
- **时延超时**：
  1. 查询调度日志确认是否存在上游卡点。
  2. 触发 `python -m inference.batch_predict --score-date <D>` 手动补跑。

## 4. 手动重跑流程
1. 确认配置：`.env`、`config/settings.yaml`、`config/labeling.yaml`。
2. 按顺序运行：
   ```bash
   python -m pipelines.ingest_raw --date <D>
   python -m pipelines.etl_standardize --date <D>
   python -m pipelines.features_build --date <D>
   python -m pipelines.score_baseline --date <D>
   python -m inference.batch_predict --score-date <D>
   ```
3. 核对 `pred_rank_daily` 与周报生成情况。

## 5. 模型回滚与恢复
1. 查看当前线上版本：`python -m training.registry list --name rank_v1`。
2. 回滚命令：`python -m training.registry demote --name rank_v1 --version <N> --to Archived`。
3. 提升备份版本：`python -m training.registry promote --name rank_v1 --version <M> --to Production`。
4. 通知业务方并记录于 `docs/release_checklist.md`。

## 6. 变更管理
- 任何配置或模型更新需提前在 `docs/release_checklist.md` 创建条目。
- 部署前完成回归测试：`pytest`、`python -m reporting.dry_run_dashboard_queries`。
- 变更完成后，更新本手册的相关章节。

## 7. 附录
- 数据库：`$DATABASE_URL`
- 存储桶：`s3://asin-data`
- 监控面板：Grafana https://grafana.example.com/d/asin

> 本文档为模板，首次上线前请补充环境特定信息并存档。
