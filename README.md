# messages-bridge

一个极小的协议桥：把 Anthropic **Messages** 请求翻成 OpenAI **Chat Completions**
发出去，再把回答翻回 Messages 返回。其余路径（`/v1/chat/completions` 等）原样透传。

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
# messages-bridge 7bc8021... listening on :8080, upstream https://api.tokenjuice.ai
```

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
| `UPSTREAM_TIMEOUT` | `10m` | 非流式请求超时 |
| `LOG_BODIES` | `false` | 是否打印转换后的上游请求体（默认关闭，避免日志把磁盘写满） |
| `BRIDGE_PORT` | `8080` | 仅 docker compose 使用：映射到宿主机的端口 |

## 路由

| 路径 | 行为 |
| --- | --- |
| `POST /v1/messages` | Messages → Chat Completions，回答再转回 Messages（含流式、tool_use/tool_calls、usage 回填） |
| `POST /v1/messages/count_tokens` | 按请求体长度粗估 token 数 |
| `GET /healthz` | 健康检查 |
| 其他所有路径 | 原样透传到 `UPSTREAM_BASE_URL`（含 `/v1/chat/completions` 流式） |

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
