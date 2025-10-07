# 推理服务接口契约（API Contract）

本文档用于约定在线推理/批量预测接口的输入、输出及错误约束，上线前请确认各方签字确认。

## 1. 服务概览
- 服务名称：Amazon ASIN Ranking API
- Base URL：`https://api.example.com/v1`
- 认证方式：API Key（`X-API-Key`）或 OAuth2 Client Credentials
- 超时时间：30s（建议客户端设置 35s 重试）

## 2. 端点定义

### 2.1 GET `/health`
- 作用：探活接口
- 响应：`200 OK`，Body：`{"status": "ok"}`

### 2.2 POST `/rank/predict`
- 请求头：
  - `Content-Type: application/json`
  - `X-API-Key: <key>`
- 请求体示例：
  ```json
  {
    "request_id": "uuid-1234",
    "items": [
      {"asin": "B000123", "site": "US", "features": {"price": 19.99, "bsr": 1200}},
      {"asin": "B000456", "site": "US", "features": {"price": 24.50, "bsr": 800}}
    ],
    "context": {"marketplace": "US", "locale": "en_US"}
  }
  ```
- 响应示例：
  ```json
  {
    "request_id": "uuid-1234",
    "generated_at": "2024-05-01T12:00:00Z",
    "items": [
      {"asin": "B000123", "score": 0.82, "rank": 1, "confidence": 0.91},
      {"asin": "B000456", "score": 0.63, "rank": 2, "confidence": 0.74}
    ]
  }
  ```
- 业务约束：
  - 每次请求最多 50 个 ASIN。
  - 若缺失 `features` 字段，将返回 `400 Bad Request`。

### 2.3 POST `/rank/batch`
- 功能：触发批量预测并返回任务 ID。
- 典型响应：`202 Accepted`，`{"job_id": "batch-20240501"}`。
- 客户端需轮询 `/rank/batch/<job_id>` 获取状态。

## 3. 错误码
| HTTP 状态码 | 错误码 | 说明 | 处理建议 |
| ----------- | ------ | ---- | -------- |
| 400 | `INVALID_PAYLOAD` | 请求格式错误或字段缺失 | 检查必填字段、字段类型 |
| 401 | `UNAUTHENTICATED` | API Key/OAuth 认证失败 | 更新凭证或联系管理员 |
| 403 | `FORBIDDEN` | 无访问权限 | 校验访问控制策略 |
| 429 | `RATE_LIMITED` | 超过 QPS/QPM 限制 | 实施退避重试或联系运营提额 |
| 500 | `INTERNAL_ERROR` | 服务内部错误 | 收集 `request_id` 并反馈给运维 |
| 504 | `TIMEOUT` | 上游耗时过长 | 检查批处理/特征查询延迟 |

## 4. 契约变更流程
1. 变更申请通过 RFC 文档提议，评审后更新本文件。
2. 重大变更需提前 2 周通知下游系统。
3. 历史版本存放于 `docs/api_contract/` 目录。

## 5. 测试与监控
- 集成测试：`pytest tests/test_inference_batch.py`
- 合约测试：使用 Postman/Newman 集合 `contracts/inference_collection.json`。
- 监控指标：成功率、P95 时延、TopN 命中率、PSI。

## 6. 审批记录
| 日期 | 版本 | 变更摘要 | 申请人 | 审批人 |
| ---- | ---- | -------- | ------ | ------ |
|      |      |          |        |        |

> 请在正式上线前完成所有字段并由相关方签署。
