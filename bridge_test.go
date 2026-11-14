package main

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

func testConfig() config {
	return config{reasoningEffort: true}
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

	body := recorder.Body.String()
	if got := recorder.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Errorf("content type = %q", got)
	}
	for _, event := range []string{
		"event: message_start",
		`"model":"client-model"`,
		"event: content_block_start",
		`"type":"text_delta","text":"Hel"`,
		`"type":"text_delta","text":"lo"`,
		`"type":"tool_use","id":"call_1","name":"Bash"`,
		`"partial_json":"{\"cmd\""`,
		`"partial_json":":\"ls\"}"`,
		"event: content_block_stop",
		"event: message_delta",
		`"stop_reason":"tool_use"`,
		`"usage":{"input_tokens":12,"output_tokens":5}`,
		"event: message_stop",
	} {
		if !strings.Contains(body, event) {
			t.Errorf("stream is missing %q\n%s", event, body)
		}
	}
	if strings.Index(body, "event: message_start") > strings.Index(body, "event: content_block_start") {
		t.Errorf("message_start must come first\n%s", body)
	}
	if strings.Count(body, "event: message_stop") != 1 {
		t.Errorf("expected exactly one message_stop\n%s", body)
	}
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
