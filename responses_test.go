package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestConvertClaudeRequestToResponses(t *testing.T) {
	var request claudeRequest
	raw := `{"model":"m","max_tokens":100,"system":"guide","output_config":{"effort":"high"},"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{"q":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":"ok"}]}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`
	if err := json.Unmarshal([]byte(raw), &request); err != nil {
		t.Fatal(err)
	}
	got, err := convertClaudeToResponses(&request, testConfig())
	if err != nil {
		t.Fatal(err)
	}
	if got.Model != "m" || got.Instructions != "guide" || got.MaxOutputTokens == nil || *got.MaxOutputTokens != 100 {
		t.Fatalf("request fields: %+v", got)
	}
	if len(got.Input) != 3 || len(got.Tools) != 1 || got.Reasoning["effort"] != "high" {
		t.Fatalf("converted request: %+v", got)
	}
	encoded, _ := json.Marshal(got)
	if !strings.Contains(string(encoded), `"type":"function_call"`) || !strings.Contains(string(encoded), `"type":"function_call_output"`) {
		t.Fatalf("tool turn missing: %s", encoded)
	}
}

func TestConvertResponsesToClaude(t *testing.T) {
	raw := `{"id":"resp_1","status":"completed","output":[{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]},{"type":"message","content":[{"type":"output_text","text":"hello"}]},{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":\"x\"}"}],"usage":{"input_tokens":7,"output_tokens":4}}`
	var response responsesOutput
	if err := json.Unmarshal([]byte(raw), &response); err != nil {
		t.Fatal(err)
	}
	got := convertResponsesToClaude(&response, "client-model")
	if got.ID != "msg_resp_1" || got.Model != "client-model" || got.StopReason != "tool_use" {
		t.Fatalf("response: %+v", got)
	}
	if len(got.Content) != 3 || got.Content[0].Type != "thinking" || got.Content[1].Text != "hello" || got.Content[2].ID != "call_1" {
		t.Fatalf("content: %+v", got.Content)
	}
	if got.Usage.InputTokens != 7 || got.Usage.OutputTokens != 4 {
		t.Fatalf("usage: %+v", got.Usage)
	}
}

func TestHandleMessagesUsesResponsesEndpoint(t *testing.T) {
	var path string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":2,"output_tokens":1}}`))
	}))
	defer upstream.Close()
	base, _ := url.Parse(upstream.URL)
	cfg := testConfig()
	cfg.upstreamBase = base
	cfg.responsesAPI = true
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
	handleMessagesResponses(recorder, request, cfg)
	if path != "/v1/responses" || recorder.Code != http.StatusOK {
		t.Fatalf("path=%q status=%d body=%s", path, recorder.Code, recorder.Body.String())
	}
	var response claudeResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if len(response.Content) != 1 || response.Content[0].Text != "ok" {
		t.Fatalf("response: %+v", response)
	}
}

func TestLoadConfigUpstreamAPIMode(t *testing.T) {
	t.Setenv("UPSTREAM_BASE_URL", "https://api.example.com")
	t.Setenv("UPSTREAM_API", "responses")
	cfg, err := loadConfig()
	if err != nil || !cfg.responsesAPI {
		t.Fatalf("responses config = %+v, err=%v", cfg, err)
	}
	t.Setenv("UPSTREAM_API", "unknown")
	if _, err := loadConfig(); err == nil {
		t.Fatal("invalid UPSTREAM_API must be rejected")
	}
}

// The switch lives in route wiring, so exercise it through the same mux main
// serves rather than calling a handler directly.
func TestNewMuxRoutesToSelectedUpstreamAPI(t *testing.T) {
	cases := []struct {
		name        string
		responses   bool
		wantPath    string
		upstreamRes string
	}{
		{
			name:        "chat completions",
			wantPath:    "/v1/chat/completions",
			upstreamRes: `{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`,
		},
		{
			name:        "responses",
			responses:   true,
			wantPath:    "/v1/responses",
			upstreamRes: `{"id":"resp_1","status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`,
		},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			var gotPath string
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(testCase.upstreamRes))
			}))
			defer upstream.Close()

			base, _ := url.Parse(upstream.URL)
			cfg := testConfig()
			cfg.upstreamBase = base
			cfg.responsesAPI = testCase.responses

			recorder := httptest.NewRecorder()
			request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
			newMux(cfg, http.NotFoundHandler()).ServeHTTP(recorder, request)

			if gotPath != testCase.wantPath {
				t.Fatalf("upstream path = %q, want %q", gotPath, testCase.wantPath)
			}
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
			}
		})
	}
}
