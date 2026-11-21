# messages-bridge

一个极小的协议桥：把 Anthropic **Messages** 请求翻成 OpenAI **Chat Completions**
发出去，再把回答翻回 Messages 返回。其余路径（`/v1/chat/completions` 等）默认原样透传；
原生 `POST /v1/responses` 可显式开启下述无状态兼容模式。

用途：让只支持 OpenAI 协议的上游（例如只接受纯净 Chat Completions 的网关）
也能给 Claude Code 之类的 Messages 客户端用；同时上游返回什么就透传什么，
不受网关自身转换逻辑影响。

## 快速开始（docker compose）

```bash
cp .env.example .env   # 填 UPSTREAM_BASE_URL（密钥不用填，由 new-api 传）/ MODEL_MAP
docker compose up -d
docker compose logs -f
```

`docker-compose.yml` 里已经带上了日志轮转（`max-size=10m`、`max-file=3`）和
`/healthz` 健康检查；想本地构建就把 `image:` 换成 `build: .`。

更新镜像要用 `docker compose pull`，只 `up -d` 会继续用本地旧镜像。
启动日志第一行带构建版本（git sha），`/healthz` 也会返回它，可以据此确认跑的是哪个版本：

```bash
docker compose logs | head -1
# messages-bridge 7bc8021... listening on :8080, upstream https://api.tokenjuice.ai, api chat_completions, proxy none
```

改用 Responses API 时要动 `.env` 里的 `UPSTREAM_API`，并且让容器真正拿到新环境变量：
只 `docker compose up -d` 不会重建容器，旧容器里的环境变量仍然生效。完整流程是

```bash
docker compose pull && docker compose up -d --force-recreate
```

启动日志里的 `api` 字段就是实际生效的协议，可据此确认开关有没有被读到。

只跑单容器：

```bash
docker run -d --name messages-bridge -p 8080:8080 \
  -e UPSTREAM_BASE_URL=https://api.example.com \
  -e UPSTREAM_API_KEY=sk-xxx \
  ghcr.io/xingguangcuican6666/messages-bridge:latest
```

或本地直接跑：

```bash
UPSTREAM_BASE_URL=https://api.example.com UPSTREAM_API_KEY=sk-xxx go run .
```

## 配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LISTEN_ADDR` | `:8080` | 监听地址 |
| `UPSTREAM_BASE_URL` | 必填 | 上游地址，如 `https://api.tokenjuice.ai`（不带 `/v1`） |
| `UPSTREAM_API_KEY` | 空 | **兜底密钥**：正常情况下留空，由 new-api 渠道把密钥放在 `Authorization` / `x-api-key` / `api-key` 里传进来并原样转发；只有调用方不带密钥时才用它 |
| `MODEL_MAP` | 空 | JSON 对象，客户端模型名 → 上游模型名，例如 `{"deepseek-v4.1-flash-1017-tj":"deepseek-ai/DeepSeek-V4.1-Flash-1017"}` |
| `PASS_REASONING_EFFORT` | `true` | 是否把 `output_config.effort` 转成 `reasoning_effort` |
| `UPSTREAM_API` | `chat_completions` | `/v1/messages` 的上游协议：`chat_completions`（默认）或 `responses`（调用 `/v1/responses`）。docker compose 部署时必须改 `.env`，并在改完后重建容器 |
| `RESPONSES_STATELESS` | `false` | 仅原生 `POST /v1/responses`：删除顶层 `previous_response_id` 和 `conversation`；客户端必须发送完整历史，不保证解除上游轮数限制。接受 true/false（忽略大小写和两端空白），其他非空值启动时报错 |
| `RESPONSES_STATELESS_DROP_HEADERS` | 空 | 仅在上述模式开启时删除显式选择的会话头，逗号分隔；只支持 `Session-Id`、`X-Session-Id`、`Conversation-Id`、`X-Conversation-Id`（忽略大小写、去重、忽略空项），其他名称即使模式关闭也会导致启动报错 |
| `UPSTREAM_TIMEOUT` | `10m` | 非流式请求超时 |
| `UPSTREAM_PROXY` | 空 | 上游代理，可写 `192.168.31.109:7890`（裸 host:port 自动按 http 处理）或 `http://host.docker.internal:7890`；留空时用容器里的 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`，都没有就直连 |
| `LOG_BODIES` | `false` | 是否打印转换后的上游请求体（默认关闭，避免日志把磁盘写满） |
| `BRIDGE_PORT` | `8080` | 仅 docker compose 使用：映射到宿主机的端口 |

## 路由

| 路径 | 行为 |
| --- | --- |
| `POST /v1/messages` | 默认 Messages → Chat Completions，设置 `UPSTREAM_API=responses` 后改走 OpenAI Responses API；回答均转回 Messages（支持非流式和流式响应） |
| `POST /v1/messages/count_tokens` | 按请求体长度粗估 token 数 |
| `GET /healthz` | 健康检查 |
| `POST /v1/responses` | 默认原样透传；`RESPONSES_STATELESS=true` 时先删除上述续链字段和显式配置的会话头，响应仍透传 |
| 其他所有路径 | 原样透传到 `UPSTREAM_BASE_URL`（含 `/v1/chat/completions` 流式） |

## 原生 Responses 无状态兼容模式

此开关用于受控比较显式会话关联信息对上游的影响，**不是上游十轮限制的已验证修复**。
`/v1/messages` 转换本来就不发送这些关联信息，但仍可能遇到上游
`session_turn_limit_exceeded`。开启本模式不会改变 Messages 转换，也不受
`UPSTREAM_API` 选用 Chat Completions 还是 Responses 影响。

启用前，客户端必须每次发送所需的**完整历史、instructions、工具调用与工具结果**。
只发送新增消息、依赖 `previous_response_id`、`conversation` 或服务端 item reference
取回历史的客户端不适用：bridge 没有会话存储，不能自动还原缺失上下文。
`previous_response_id` 是上游生成的引用，不能随机伪造来充当新会话。

在 `.env` 中设置：

```dotenv
RESPONSES_STATELESS=true
# 可选，只有确认上游使用这些头关联会话时才选填；留空不删任何头
RESPONSES_STATELESS_DROP_HEADERS=X-Session-Id,X-Conversation-Id
```

首版只支持配置表中的四个精确 header 名称，不支持任意名称或通配符；该名单不表示
已证实上游识别这些头。需要其他头时应先确认其语义，再扩展实现。认证、Cookie、
租户和渠道路由头不在删除范围内。

容器须运行包含本功能的新版本镜像，并重新创建以加载环境变量，例如
`docker compose pull && docker compose up -d --force-recreate`（使用已发布镜像时）。
启动日志的 `responses_stateless true/false` 表示实际生效状态。恢复默认只需设置
`RESPONSES_STATELESS=false` 后重新创建容器；header 列表单独配置不生效。

启用后的请求侧行为：

- 只处理原生、精确匹配的 `POST /v1/responses`；只删**顶层**的 `previous_response_id`
  和 `conversation`，以及显式选中的会话头。不改模型、不截断 `input`，不动嵌套工具 ID、
  instructions、metadata、user、store 或未知字段，不自动设置 `store=false`。
- 没有上述顶层字段时保留原始 body 字节；实际删除字段时会重新编码顶层 JSON 对象。
- 必须是单个非 null JSON 对象，且不能有重复顶层键；非法 JSON、其他根类型、尾随文档
  或读取失败返回 HTTP 400。body 上限为 64 MiB，超限返回 413，不会截断后转发。
- 不支持非 identity 的 `Content-Encoding` 或请求 trailers，返回 415。需要 body 签名的
  请求不适用：实际重写时若含 `Content-MD5`、`Digest`、`Content-Digest`、`Repr-Digest`、
  `Signature` 或 `Signature-Input`，返回 415，而不是转发失效摘要。这不是通用签名验证器。
- 上述本地拒绝都发生在调用上游前；默认关闭时不新增这些解析与限制。
- 不生成 response/session ID，不改响应 ID，不因十轮错误换标识或自动重试。
  普通 JSON、HTTP 错误和 SSE 都沿用原代理透传，不修补 Messages 转换中的流式错误行为。

## 走代理

上游要过代理时设 `UPSTREAM_PROXY`，转换请求和透传请求共用同一个 transport：

```bash
# .env：宿主机开了局域网访问，直接写 IP
UPSTREAM_PROXY=192.168.31.109:7890
```

两种写法都行：

- `192.168.31.109:7890` —— 宿主机 LAN IP，代理允许局域网访问时用这个（不写 `http://` 会自动补上）
- `http://host.docker.internal:7890` —— 代理只监听 `127.0.0.1` 时用这个；
  `docker-compose.yml` 里已经加了 `host.docker.internal:host-gateway`

注意容器里的 `127.0.0.1` 是容器自己，填 `127.0.0.1:7890` 连不上。

不填 `UPSTREAM_PROXY` 时沿用容器里的 `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY`
（Go 的默认行为），都没有就直连。启动日志会打印实际生效的代理：

```
messages-bridge f2e2b94 listening on :8080, upstream https://api.tokenjuice.ai, proxy http://host.docker.internal:7890
```

## 在 new-api 里接入

1. 新建渠道，类型选 **OpenAI**，Base URL 填 `http://messages-bridge:8080`，
   密钥填**上游真实密钥**——桥服务会把 new-api 发来的这个密钥原样转给上游，
   本服务自己不保存任何密钥。
2. 开启该渠道的**透传**（`pass_through_body_enabled`），让 `/v1/messages` 与 `/v1/chat/completions` 都按原生格式直发到本服务。
3. 客户端（Claude Code）把 Base URL 指向 new-api 即可。

> 透传下 new-api 不再改写请求体，转换与工具调用映射全部由本服务负责；
> 计费所需的 usage 由本服务在 Messages 响应/流里回填。
>
> 换密钥、加渠道、按渠道做密钥轮换，都只改 new-api 里的渠道配置，
> 桥服务不用重启也不用改环境变量。

## 转换约定

- 请求只输出 Chat Completions 规范字段：`model` / `messages` / `max_tokens` / `temperature` / `top_p` / `stop` / `stream` / `stream_options`(仅流式) / `tools` / `tool_choice` / `reasoning_effort`。
- 不发送 `top_k`、`cache_control`、`$schema` 等扩展字段；纯文本内容写成字符串。
- 工具 schema 会做两处净化，因为上游对这两类内容是"整单拒绝"（返回 `{"message":"Invalid request"}`）：
  - 删除空 `enum` 数组（Claude Code 会给没有可选值的参数发 `"enum": []`）；
  - 删除上游正则引擎不支持的 `pattern`（前瞻/后顾/反向引用/`\p{...}` 等），简单正则原样保留。
- `tool_result` 块转成 `role:"tool"` 消息，`tool_use` 块转成 `tool_calls`。
- 响应侧把 `finish_reason` 映射为 `stop_reason`（`tool_calls`→`tool_use`，`length`→`max_tokens`），`prompt_tokens`/`completion_tokens` 回填为 `input_tokens`/`output_tokens`。
