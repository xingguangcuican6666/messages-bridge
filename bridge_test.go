package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

func testConfig() config {
	return config{reasoningEffort: true, timeout: 30 * time.Second}
}

func TestConvertClaudeRequestToPureOpenAI(t *testing.T) {
	raw := `{
		"model": "deepseek-v4.1-flash-1017-tj",
		"max_tokens": 32000,
		"system": [{"type":"text","text":"You are Claude Code."},{"type":"text","text":"# Environment"}],
		"output_config": {"effort": "high"},
		"stop_sequences": ["STOP"],
		"stream": true,
		"tool_choice": {"type": "auto"},
		"tools": [{"name":"Bash","description":"run","input_schema":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}],
		"messages": [
			{"role":"user","content":[{"type":"text","text":"run it"}]},
			{"role":"assistant","content":[{"type":"text","text":"running"},{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"cmd":"ls"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"text","text":"a.txt"}]},{"type":"text","text":"now summarize"}]}
		]
	}`

	var request claudeRequest
	if err := json.Unmarshal([]byte(raw), &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}

	converted, err := convertClaudeToOpenAI(&request, testConfig())
	if err != nil {
		t.Fatalf("convert: %v", err)
	}

	if converted.Model != "deepseek-v4.1-flash-1017-tj" {
		t.Errorf("model = %q", converted.Model)
	}
	if converted.MaxTokens == nil || *converted.MaxTokens != 32000 {
		t.Errorf("max_tokens not preserved")
	}
	if converted.ReasoningEffort != "high" {
		t.Errorf("reasoning_effort = %q", converted.ReasoningEffort)
	}
	if converted.Stop != "STOP" {
		t.Errorf("stop = %v", converted.Stop)
	}
	if !converted.Stream || converted.StreamOptions == nil || !converted.StreamOptions.IncludeUsage {
		t.Errorf("stream options missing")
	}
	if converted.ToolChoice != "auto" {
		t.Errorf("tool_choice = %v", converted.ToolChoice)
	}

	if len(converted.Messages) != 5 {
		t.Fatalf("messages = %d, want 5: %+v", len(converted.Messages), converted.Messages)
	}
	if converted.Messages[0].Role != "system" || converted.Messages[0].Content != "You are Claude Code.# Environment" {
		t.Errorf("system message = %+v", converted.Messages[0])
	}
	if converted.Messages[1].Role != "user" || converted.Messages[1].Content != "run it" {
		t.Errorf("user message = %+v", converted.Messages[1])
	}
	assistant := converted.Messages[2]
	if assistant.Role != "assistant" || assistant.Content != "running" {
		t.Errorf("assistant message = %+v", assistant)
	}
	if len(assistant.ToolCalls) != 1 || assistant.ToolCalls[0].ID != "toolu_1" || assistant.ToolCalls[0].Function.Arguments != `{"cmd":"ls"}` {
		t.Errorf("assistant tool calls = %+v", assistant.ToolCalls)
	}
	toolMessage := converted.Messages[3]
	if toolMessage.Role != "tool" || toolMessage.ToolCallID != "toolu_1" || toolMessage.Content != "a.txt" {
		t.Errorf("tool message = %+v", toolMessage)
	}
	if converted.Messages[4].Role != "user" || converted.Messages[4].Content != "now summarize" {
		t.Errorf("trailing user message = %+v", converted.Messages[4])
	}

	if len(converted.Tools) != 1 {
		t.Fatalf("tools = %d", len(converted.Tools))
	}
	parameters, ok := converted.Tools[0].Function.Parameters.(map[string]any)
	if !ok {
		t.Fatalf("parameters type = %T", converted.Tools[0].Function.Parameters)
	}
	if _, exists := parameters["$schema"]; exists {
		t.Errorf("$schema must be stripped from tool parameters")
	}
	if parameters["type"] != "object" {
		t.Errorf("schema type = %v", parameters["type"])
	}

	encoded, err := json.Marshal(converted)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, forbidden := range []string{"top_k", "cache_control", "verbosity", "$schema"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Errorf("emitted body must not contain %q: %s", forbidden, encoded)
		}
	}
}

func TestConvertToolChoiceByName(t *testing.T) {
	choice := convertToolChoice(json.RawMessage(`{"type":"tool","name":"Bash"}`))
	mapped, ok := choice.(map[string]any)
	if !ok {
		t.Fatalf("tool choice type = %T", choice)
	}
	function := mapped["function"].(map[string]string)
	if mapped["type"] != "function" || function["name"] != "Bash" {
		t.Errorf("tool choice = %v", choice)
	}
	if convertToolChoice(json.RawMessage(`{"type":"any"}`)) != "required" {
		t.Errorf("any must map to required")
	}
}

func TestConvertOpenAIResponseToClaude(t *testing.T) {
	raw := `{
		"id": "chatcmpl-123",
		"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
		"choices": [{"message": {"role":"assistant","content":"hi","tool_calls":[{"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{\"cmd\":\"ls\"}"}}]},"finish_reason":"tool_calls"}],
		"usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
	}`
	var completion openAIResponse
	if err := json.Unmarshal([]byte(raw), &completion); err != nil {
		t.Fatalf("decode: %v", err)
	}

	response := convertOpenAIToClaude(&completion, "deepseek-v4.1-flash-1017-tj")
	if response.Model != "deepseek-v4.1-flash-1017-tj" {
		t.Errorf("model must echo the client model, got %q", response.Model)
	}
	if response.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q", response.StopReason)
	}
	if response.Usage.InputTokens != 11 || response.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v", response.Usage)
	}
	if len(response.Content) != 2 {
		t.Fatalf("content blocks = %+v", response.Content)
	}
	if response.Content[0].Type != "text" || response.Content[0].Text != "hi" {
		t.Errorf("text block = %+v", response.Content[0])
	}
	tool := response.Content[1]
	if tool.Type != "tool_use" || tool.ID != "call_1" || tool.Name != "Bash" {
		t.Errorf("tool block = %+v", tool)
	}
	input, ok := tool.Input.(map[string]any)
	if !ok || input["cmd"] != "ls" {
		t.Errorf("tool input = %v", tool.Input)
	}
}

func TestTranslateStreamEmitsAnthropicEvents(t *testing.T) {
	upstream := strings.Join([]string{
		`data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant","content":"Hel"}}]}`,
		"",
		`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"lo"}}]}`,
		"",
		`data: {"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Bash","arguments":"{\"cmd\""}}]}}]}`,
		"",
		`data: {"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"ls\"}"}}]},"finish_reason":"tool_calls"}]}`,
		"",
		`data: {"id":"chatcmpl-1","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")

	recorder := httptest.NewRecorder()
	if err := translateStream(recorder, strings.NewReader(upstream), "client-model"); err != nil {
		t.Fatalf("translate: %v", err)
	}

	if got := recorder.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Errorf("content type = %q", got)
	}
	events := parseSSE(t, recorder.Body.String())

	wantNames := []string{
		"message_start",
		"content_block_start", "content_block_delta", "content_block_delta",
		"content_block_stop",
		"content_block_start", "content_block_delta", "content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	names := make([]string, 0, len(events))
	for _, event := range events {
		names = append(names, event.name)
	}
	if strings.Join(names, ",") != strings.Join(wantNames, ",") {
		t.Fatalf("event sequence = %v\nwant %v\nraw:\n%s", names, wantNames, recorder.Body.String())
	}

	started := events[0].data["message"].(map[string]any)
	if started["model"] != "client-model" {
		t.Errorf("message_start model = %v", started["model"])
	}

	var textDeltas []string
	var partialJSON []string
	var toolStart map[string]any
	for _, event := range events {
		switch event.name {
		case "content_block_start":
			block := event.data["content_block"].(map[string]any)
			if block["type"] == "tool_use" {
				toolStart = block
			}
		case "content_block_delta":
			delta := event.data["delta"].(map[string]any)
			switch delta["type"] {
			case "text_delta":
				textDeltas = append(textDeltas, delta["text"].(string))
			case "input_json_delta":
				partialJSON = append(partialJSON, delta["partial_json"].(string))
			}
		}
	}
	if strings.Join(textDeltas, "") != "Hello" {
		t.Errorf("text deltas = %v", textDeltas)
	}
	if strings.Join(partialJSON, "") != `{"cmd":"ls"}` {
		t.Errorf("tool argument fragments = %v", partialJSON)
	}
	if toolStart == nil || toolStart["id"] != "call_1" || toolStart["name"] != "Bash" {
		t.Errorf("tool_use block start = %v", toolStart)
	}

	delta := events[len(events)-2].data
	stop := delta["delta"].(map[string]any)
	if stop["stop_reason"] != "tool_use" {
		t.Errorf("stop_reason = %v", stop["stop_reason"])
	}
	usage := delta["usage"].(map[string]any)
	if usage["input_tokens"] != float64(12) || usage["output_tokens"] != float64(5) {
		t.Errorf("usage = %v", usage)
	}
}

type sseEvent struct {
	name string
	data map[string]any
}

func parseSSE(t *testing.T, body string) []sseEvent {
	t.Helper()
	var events []sseEvent
	for _, line := range strings.Split(body, "\n") {
		if name, ok := strings.CutPrefix(line, "event: "); ok {
			events = append(events, sseEvent{name: name})
			continue
		}
		payload, ok := strings.CutPrefix(line, "data: ")
		if !ok || len(events) == 0 {
			continue
		}
		var data map[string]any
		if err := json.Unmarshal([]byte(payload), &data); err != nil {
			t.Fatalf("event %q has an unparsable payload: %v", events[len(events)-1].name, err)
		}
		events[len(events)-1].data = data
	}
	return events
}

func TestUpstreamMessageExtraction(t *testing.T) {
	cases := map[string]string{
		`{"error":{"message":"Invalid request"}}`: "Invalid request",
		`{"message":"Bad Request"}`:               "Bad Request",
		`{"error":"Invalid request"}`:             "Invalid request",
	}
	for body, want := range cases {
		if got := upstreamMessage([]byte(body)); got != want {
			t.Errorf("upstreamMessage(%s) = %q, want %q", body, got, want)
		}
	}
}

func TestHandleMessagesEndToEnd(t *testing.T) {
	var receivedBody []byte
	var receivedAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("upstream path = %q", r.URL.Path)
		}
		receivedAuth = r.Header.Get("Authorization")
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-9","choices":[{"message":{"role":"assistant","content":"hi there"},"finish_reason":"stop"}],"usage":{"prompt_tokens":21,"completion_tokens":3,"total_tokens":24}}`))
	}))
	defer upstream.Close()

	base, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parse upstream: %v", err)
	}
	cfg := testConfig()
	cfg.upstreamBase = base

	requestBody := `{"model":"client-model","max_tokens":128,"top_k":40,"system":"be brief","messages":[{"role":"user","content":[{"type":"text","text":"hello","cache_control":{"type":"ephemeral"}}]}]}`
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(requestBody))
	request.Header.Set("Authorization", "Bearer sk-test")
	recorder := httptest.NewRecorder()

	handleMessages(recorder, request, cfg)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if receivedAuth != "Bearer sk-test" {
		t.Errorf("forwarded auth = %q", receivedAuth)
	}
	for _, forbidden := range []string{"top_k", "cache_control"} {
		if strings.Contains(string(receivedBody), forbidden) {
			t.Errorf("upstream body must not contain %q: %s", forbidden, receivedBody)
		}
	}
	if !strings.Contains(string(receivedBody), `"role":"system"`) || !strings.Contains(string(receivedBody), `"content":"hello"`) {
		t.Errorf("upstream body lost the converted messages: %s", receivedBody)
	}

	var response claudeResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Type != "message" || response.Model != "client-model" || response.StopReason != "end_turn" {
		t.Errorf("response header fields = %+v", response)
	}
	if len(response.Content) != 1 || response.Content[0].Text != "hi there" {
		t.Errorf("response content = %+v", response.Content)
	}
	if response.Usage.InputTokens != 21 || response.Usage.OutputTokens != 3 {
		t.Errorf("response usage = %+v", response.Usage)
	}
}

func TestHandleMessagesRelaysUpstreamError(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Invalid request"}}`))
	}))
	defer upstream.Close()

	base, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parse upstream: %v", err)
	}
	cfg := testConfig()
	cfg.upstreamBase = base

	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`))
	recorder := httptest.NewRecorder()

	handleMessages(recorder, request, cfg)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", recorder.Code)
	}
	var payload struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode error body: %v", err)
	}
	if payload.Type != "error" || payload.Error.Type != "invalid_request_error" || payload.Error.Message != "Invalid request" {
		t.Errorf("error payload = %+v", payload)
	}
}

func TestHandleMessagesStreamsAnthropicEvents(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-2\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"id\":\"chatcmpl-2\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":1}}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	base, err := url.Parse(upstream.URL)
	if err != nil {
		t.Fatalf("parse upstream: %v", err)
	}
	cfg := testConfig()
	cfg.upstreamBase = base

	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"client-model","max_tokens":8,"stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	recorder := httptest.NewRecorder()

	handleMessages(recorder, request, cfg)

	events := parseSSE(t, recorder.Body.String())
	if len(events) < 5 {
		t.Fatalf("events = %v", events)
	}
	if events[0].name != "message_start" || events[len(events)-1].name != "message_stop" {
		t.Fatalf("event sequence = %v", events)
	}
	usage := events[len(events)-2].data["usage"].(map[string]any)
	if usage["input_tokens"] != float64(5) || usage["output_tokens"] != float64(1) {
		t.Errorf("stream usage = %v", usage)
	}
}

func TestUpstreamKeyComesFromCaller(t *testing.T) {
	cfg := testConfig()
	cfg.upstreamKey = "configured-fallback"

	cases := []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{"bearer wins over configured key", map[string]string{"Authorization": "Bearer sk-from-newapi"}, "sk-from-newapi"},
		{"x-api-key is forwarded", map[string]string{"x-api-key": "sk-anthropic-style"}, "sk-anthropic-style"},
		{"api-key is forwarded", map[string]string{"api-key": "sk-azure-style"}, "sk-azure-style"},
		{"configured key is the fallback", nil, "configured-fallback"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}"))
			for name, value := range tc.headers {
				request.Header.Set(name, value)
			}
			if got := resolveUpstreamKey(request, cfg); got != tc.want {
				t.Errorf("key = %q, want %q", got, tc.want)
			}
		})
	}

	// Without a configured key the caller's credential is still forwarded, and a
	// caller that sends nothing leaves the upstream call unauthenticated.
	cfg.upstreamKey = ""
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}"))
	request.Header.Set("Authorization", "Bearer sk-from-newapi")
	if got := resolveUpstreamKey(request, cfg); got != "sk-from-newapi" {
		t.Errorf("key = %q", got)
	}
	bare := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}"))
	if got := resolveUpstreamKey(bare, cfg); got != "" {
		t.Errorf("key = %q, want empty", got)
	}
}

func TestToolSchemaDropsEmptyEnum(t *testing.T) {
	// Claude Code emits "enum": [] for arguments that lost every allowed value,
	// and the upstream rejects the entire request with a bare "Invalid request".
	raw := `{
		"model": "m",
		"max_tokens": 64,
		"tools": [{
			"name": "Bash",
			"input_schema": {
				"$schema": "http://json-schema.org/draft-07/schema#",
				"type": "object",
				"properties": {
					"command": {"type": "string"},
					"effort": {"type": "string", "enum": []},
					"nested": {
						"$schema": "http://json-schema.org/draft-07/schema#",
						"type": "object",
						"properties": {"mode": {"enum": []}, "level": {"enum": ["low", "high"]}}
					}
				}
			}
		}],
		"messages": [{"role": "user", "content": "hi"}]
	}`

	var request claudeRequest
	if err := json.Unmarshal([]byte(raw), &request); err != nil {
		t.Fatalf("decode request: %v", err)
	}
	converted, err := convertClaudeToOpenAI(&request, testConfig())
	if err != nil {
		t.Fatalf("convert: %v", err)
	}
	encoded, err := json.Marshal(converted.Tools[0].Function.Parameters)
	if err != nil {
		t.Fatalf("marshal parameters: %v", err)
	}
	body := string(encoded)
	if strings.Contains(body, `"enum":[]`) {
		t.Errorf("empty enum must be stripped: %s", body)
	}
	if strings.Contains(body, "$schema") {
		t.Errorf("nested $schema must be stripped: %s", body)
	}
	if !strings.Contains(body, `"enum":["low","high"]`) {
		t.Errorf("non-empty enum must survive: %s", body)
	}
}
