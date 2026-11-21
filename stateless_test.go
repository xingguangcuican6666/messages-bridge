package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestResponsesStatelessConfig(t *testing.T) {
	t.Setenv("UPSTREAM_BASE_URL", "https://upstream.invalid")
	t.Setenv("UPSTREAM_API", "chat_completions")
	t.Setenv("UPSTREAM_PROXY", "")
	t.Setenv("UPSTREAM_TIMEOUT", "")
	t.Setenv("MODEL_MAP", "")
	cases := []struct {
		name, enabled, headers string
		wantEnabled            bool
		wantHeaders            []string
		wantError              bool
	}{
		{name: "defaults"},
		{name: "whitespace", enabled: "  "},
		{name: "enabled", enabled: " TrUe ", wantEnabled: true},
		{name: "disabled", enabled: " FALSE "},
		{name: "numeric", enabled: "1", wantError: true},
		{name: "typo", enabled: "yes", wantError: true},
		{name: "normalize", enabled: "true", headers: " x-session-id, SESSION-id, ,X-Session-ID,conversation-id,x-conversation-id,", wantEnabled: true, wantHeaders: []string{"X-Session-Id", "Session-Id", "Conversation-Id", "X-Conversation-Id"}},
		{name: "header list alone", headers: "X-Session-Id", wantHeaders: []string{"X-Session-Id"}},
	}
	for _, name := range []string{"Authorization", "Proxy-Authorization", "X-Api-Key", "Api-Key", "Cookie", "Host", "Content-Length", "Connection", "OpenAI-Project", "X-Route", "X-*", "X-Session-Id\r\nAuthorization"} {
		cases = append(cases, struct {
			name, enabled, headers string
			wantEnabled            bool
			wantHeaders            []string
			wantError              bool
		}{name: "reject " + name, headers: name, wantError: true})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("RESPONSES_STATELESS", tc.enabled)
			t.Setenv("RESPONSES_STATELESS_DROP_HEADERS", tc.headers)
			cfg, err := loadConfig()
			if cfg.transport != nil {
				defer cfg.transport.CloseIdleConnections()
			}
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError = %t", err, tc.wantError)
			}
			if !tc.wantError && (cfg.responsesStateless != tc.wantEnabled || !reflect.DeepEqual(cfg.responsesStatelessHeaders, tc.wantHeaders)) {
				t.Fatalf("enabled=%t headers=%v", cfg.responsesStateless, cfg.responsesStatelessHeaders)
			}
		})
	}
}

func statelessTestMux(t *testing.T, cfg config, upstream http.Handler) http.Handler {
	t.Helper()
	server := httptest.NewServer(upstream)
	t.Cleanup(server.Close)
	cfg.upstreamBase, _ = url.Parse(server.URL)
	// These tests must never use a proxy inherited from the host environment.
	cfg.transport = &http.Transport{Proxy: nil}
	t.Cleanup(cfg.transport.CloseIdleConnections)
	proxy, err := newPassthroughProxy(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return newMux(cfg, proxy)
}

func TestResponsesStatelessPreservesHistoryAndFields(t *testing.T) {
	var input []string
	for i := 0; i < 12; i++ {
		input = append(input,
			fmt.Sprintf(`{"role":"user","content":"turn %d"}`, i),
			fmt.Sprintf(`{"type":"function_call","call_id":"call_%d","name":"lookup","arguments":"{}"}`, i),
			fmt.Sprintf(`{"type":"function_call_output","call_id":"call_%d","output":"result %d"}`, i, i),
			fmt.Sprintf(`{"role":"assistant","content":"answer %d"}`, i))
	}
	body := `{"previous_response_id":"resp_old","conversation":{"id":"conv_old"},"model":"client-model","input":[` + strings.Join(input, ",") + `],"instructions":"guide","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"temperature":0.3,"top_p":0.9,"store":true,"user":"user-1","metadata":{"previous_response_id":"nested","conversation":"keep"},"extension":{"n":900719925474099312345678901234567890},"stream":false}`
	var before map[string]json.RawMessage
	if err := json.Unmarshal([]byte(body), &before); err != nil {
		t.Fatal(err)
	}
	cfg := testConfig()
	cfg.responsesStateless = true
	cfg.responsesStatelessHeaders = []string{"X-Session-Id", "Conversation-Id"}
	cfg.modelMap = map[string]string{"client-model": "must-not-map-native"}
	for _, api := range []bool{false, true} {
		t.Run(fmt.Sprintf("responsesAPI=%t", api), func(t *testing.T) {
			cfg.responsesAPI = api
			calls := 0
			handler := statelessTestMux(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				got, err := io.ReadAll(r.Body)
				if err != nil {
					t.Error(err)
				}
				var after map[string]json.RawMessage
				if err := json.Unmarshal(got, &after); err != nil {
					t.Error(err)
				}
				if len(after) != len(before)-2 {
					t.Errorf("field count=%d, want=%d", len(after), len(before)-2)
				}
				for name, value := range before {
					if name == "previous_response_id" || name == "conversation" {
						if _, exists := after[name]; exists {
							t.Errorf("retained top-level %s", name)
						}
						continue
					}
					if !bytes.Equal(after[name], value) {
						t.Errorf("%s changed: %s != %s", name, after[name], value)
					}
				}
				if r.URL.Path != "/v1/responses" || r.URL.RawQuery != "q=a%2Fb&repeat=1&repeat=2" {
					t.Errorf("URL changed: %s", r.URL)
				}
				for _, name := range cfg.responsesStatelessHeaders {
					if r.Header.Get(name) != "" {
						t.Errorf("retained header %s", name)
					}
				}
				for name, value := range map[string]string{"Authorization": "Bearer caller", "X-Api-Key": "caller-api", "Api-Key": "other-api", "Cookie": "keep=1", "Session-Id": "not-selected", "X-Conversation-Id": "also-not-selected", "X-Other": "keep", "OpenAI-Project": "project", "Accept": "application/json"} {
					if r.Header.Get(name) != value {
						t.Errorf("header %s=%q, want %q", name, r.Header.Get(name), value)
					}
				}
				if r.ContentLength != int64(len(got)) || len(r.TransferEncoding) != 0 {
					t.Errorf("length=%d transfer=%v body=%d", r.ContentLength, r.TransferEncoding, len(got))
				}
				_, _ = io.WriteString(w, `{"id":"resp_original","status":"completed"}`)
			}))
			request := httptest.NewRequest(http.MethodPost, "/v1/responses?q=a%2Fb&repeat=1&repeat=2", strings.NewReader(body))
			for name, value := range map[string]string{"X-Session-Id": "drop", "Conversation-Id": "drop", "Authorization": "Bearer caller", "X-Api-Key": "caller-api", "Api-Key": "other-api", "Cookie": "keep=1", "Session-Id": "not-selected", "X-Conversation-Id": "also-not-selected", "X-Other": "keep", "OpenAI-Project": "project", "Accept": "application/json"} {
				request.Header.Set(name, value)
			}
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != 200 || recorder.Body.String() != `{"id":"resp_original","status":"completed"}` || calls != 1 {
				t.Fatalf("status=%d calls=%d body=%s", recorder.Code, calls, recorder.Body)
			}
			if request.Header.Get("X-Session-Id") != "drop" {
				t.Fatal("original request headers mutated")
			}
		})
	}
}

type statelessReadCloser struct {
	io.Reader
	closed bool
}

func (r *statelessReadCloser) Close() error { r.closed = true; return nil }

func TestResponsesStatelessRequestBookkeeping(t *testing.T) {
	for _, body := range []string{`{"input":"hello","previous_response_id":null}`, " { \n\"input\": \"unchanged\", \"number\": 9007199254740993 } \n"} {
		t.Run(body, func(t *testing.T) {
			cfg := config{responsesStateless: true}
			request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body))
			originalBody := &statelessReadCloser{Reader: strings.NewReader(body)}
			request.Body = originalBody
			request.Header.Set("Content-Length", "999")
			request.Header.Set("Transfer-Encoding", "chunked")
			request.TransferEncoding = []string{"chunked"}
			request.ContentLength = -1
			request.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(body)), nil }
			calls := 0
			next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				got, _ := io.ReadAll(r.Body)
				if !originalBody.closed || r == request || r.Context() != request.Context() {
					t.Error("body ownership, request cloning or context changed")
				}
				if r.ContentLength != int64(len(got)) || r.Header.Get("Content-Length") != "" || r.Header.Get("Transfer-Encoding") != "" || len(r.TransferEncoding) != 0 || r.GetBody != nil {
					t.Errorf("inconsistent forwarded request: length=%d headers=%v transfer=%v getBody=%t", r.ContentLength, r.Header, r.TransferEncoding, r.GetBody != nil)
				}
				if !strings.Contains(body, "previous_response_id") && string(got) != body {
					t.Fatalf("no-op body changed: %q", got)
				}
				w.WriteHeader(204)
			})
			recorder := httptest.NewRecorder()
			withResponsesStateless(next, cfg).ServeHTTP(recorder, request)
			if recorder.Code != 204 || calls != 1 || request.Header.Get("Content-Length") != "999" || request.GetBody == nil {
				t.Fatalf("status=%d calls=%d or original request mutated", recorder.Code, calls)
			}
		})
	}
}

func TestResponsesStatelessNonTargetsDoNotReadBody(t *testing.T) {
	cases := []struct {
		enabled      bool
		method, path string
	}{
		{false, "POST", "/v1/responses"},
		{true, "GET", "/v1/responses"},
		{true, "DELETE", "/v1/responses"},
		{true, "POST", "/v1/responses/"},
		{true, "POST", "/v1/responses/resp_1"},
		{true, "POST", "/v1/chat/completions"},
		{true, "POST", "/v1/messages"},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%t %s %s", tc.enabled, tc.method, tc.path), func(t *testing.T) {
			body := &statelessReadCloser{Reader: statelessErrorReader{}}
			request := httptest.NewRequest(tc.method, tc.path, nil)
			request.Body = body
			request.Header.Set("Content-Encoding", "gzip")
			request.Header.Set("X-Session-Id", "keep")
			calls := 0
			next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r != request || r.Body != body || body.closed || r.Header.Get("X-Session-Id") != "keep" {
					t.Fatal("non-target request was inspected or mutated")
				}
				w.WriteHeader(204)
			})
			recorder := httptest.NewRecorder()
			withResponsesStateless(next, config{responsesStateless: tc.enabled, responsesStatelessHeaders: []string{"X-Session-Id"}}).ServeHTTP(recorder, request)
			if recorder.Code != 204 || calls != 1 {
				t.Fatalf("status=%d calls=%d", recorder.Code, calls)
			}
		})
	}
}

type statelessErrorReader struct{}

func (statelessErrorReader) Read([]byte) (int, error) { return 0, errors.New("read must fail") }

func TestResponsesStatelessRejectsUnsafeBodies(t *testing.T) {
	cases := []struct {
		name, body, header, value string
		status                    int
	}{
		{name: "empty", status: 400},
		{name: "null", body: "null", status: 400},
		{name: "array", body: "[]", status: 400},
		{name: "string", body: `"text"`, status: 400},
		{name: "number", body: "42", status: 400},
		{name: "boolean", body: "true", status: 400},
		{name: "truncated", body: `{"input":`, status: 400},
		{name: "unclosed", body: `{"input":1`, status: 400},
		{name: "wrong close", body: `{"input":1]`, status: 400},
		{name: "extra object", body: `{} {}`, status: 400},
		{name: "extra null", body: `{} null`, status: 400},
		{name: "garbage", body: `{} garbage`, status: 400},
		{name: "trailing comma", body: `{"input":1,}`, status: 400},
		{name: "duplicate input", body: `{"input":1,"input":2}`, status: 400},
		{name: "escaped duplicate", body: `{"input":1,"input":2}`, status: 400},
		{name: "gzip", body: `{}`, header: "Content-Encoding", value: "gzip", status: 415},
		{name: "encoding list", body: `{}`, header: "Content-Encoding", value: "identity, gzip", status: 415},
		{name: "trailer header", body: `{}`, header: "Trailer", value: "Digest", status: 415},
		{name: "read error", status: 400},
		{name: "trailer field", body: `{}`, status: 415},
	}
	for _, name := range []string{"Content-MD5", "Digest", "Content-Digest", "Repr-Digest", "Signature", "Signature-Input"} {
		cases = append(cases, struct {
			name, body, header, value string
			status                    int
		}{name: name, body: `{"input":[],"conversation":"conv_1"}`, header: name, value: "synthetic", status: 415})
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			request := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(tc.body))
			if tc.header != "" {
				request.Header.Set(tc.header, tc.value)
			}
			if tc.name == "trailer field" {
				request.Trailer = http.Header{"Digest": nil}
			}
			original := &statelessReadCloser{Reader: request.Body}
			if tc.name == "read error" {
				original.Reader = statelessErrorReader{}
			}
			request.Body = original
			calls := 0
			next := http.HandlerFunc(func(http.ResponseWriter, *http.Request) { calls++ })
			recorder := httptest.NewRecorder()
			withResponsesStateless(next, config{responsesStateless: true}).ServeHTTP(recorder, request)
			if recorder.Code != tc.status || calls != 0 || !strings.Contains(recorder.Body.String(), `"type":"invalid_request_error"`) {
				t.Fatalf("status=%d calls=%d body=%s", recorder.Code, calls, recorder.Body)
			}
			if tc.status == 400 && !original.closed {
				t.Fatal("consumed body not closed")
			}
		})
	}
}

func TestResponsesStatelessRejectsUndeclaredTrailers(t *testing.T) {
	// Go only learns about an undeclared trailer while reading the chunked body.
	request, err := http.ReadRequest(bufio.NewReader(strings.NewReader("POST /v1/responses HTTP/1.1\r\nHost: bridge.invalid\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\nDigest: synthetic\r\n\r\n")))
	if err != nil {
		t.Fatal(err)
	}
	if len(request.Trailer) != 0 {
		t.Fatal("test requires an initially undeclared trailer")
	}
	calls := 0
	recorder := httptest.NewRecorder()
	withResponsesStateless(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		calls++
	}), config{responsesStateless: true}).ServeHTTP(recorder, request)
	if recorder.Code != 415 || calls != 0 {
		t.Fatalf("status=%d calls=%d body=%s", recorder.Code, calls, recorder.Body)
	}
}

func TestResponsesStatelessBodySizeBoundary(t *testing.T) {
	for _, extra := range []int{0, 1} {
		t.Run(fmt.Sprintf("extra=%d", extra), func(t *testing.T) {
			// A huge whitespace suffix avoids building a second 64 MiB JSON value.
			body := "{}" + strings.Repeat(" ", maxRequestBytes-2+extra)
			request := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
			calls := 0
			next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.ContentLength != maxRequestBytes {
					t.Errorf("length=%d", r.ContentLength)
				}
				w.WriteHeader(204)
			})
			recorder := httptest.NewRecorder()
			withResponsesStateless(next, config{responsesStateless: true}).ServeHTTP(recorder, request)
			if extra == 0 && (recorder.Code != 204 || calls != 1) || extra == 1 && (recorder.Code != 413 || calls != 0) {
				t.Fatalf("status=%d calls=%d", recorder.Code, calls)
			}
		})
	}
}

func TestResponsesStatelessNoopWithDigestAndIdentity(t *testing.T) {
	body := " {\"input\":{\"nested\":1,\"nested\":2}} \n"
	request := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(body))
	request.Header.Set("Content-Encoding", " Identity ")
	request.Header.Set("Content-Digest", "unchanged")
	recorder := httptest.NewRecorder()
	withResponsesStateless(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got, _ := io.ReadAll(r.Body)
		if string(got) != body || r.Header.Get("Content-Digest") != "unchanged" {
			t.Error("no-op body or digest changed")
		}
		w.WriteHeader(204)
	}), config{responsesStateless: true}).ServeHTTP(recorder, request)
	if recorder.Code != 204 {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body)
	}
}

func TestResponsesStatelessProxyCredentialsAndResponses(t *testing.T) {
	cases := []struct {
		name, body, contentType string
		status                  int
	}{
		{"json", `{"id":"resp_original","status":"completed","extension":1}`, "application/json", 200},
		{"turn limit", `{"error":{"code":"session_turn_limit_exceeded","message":"This conversation has reached 10 turns."}}`, "application/json", 400},
		{"SSE failed", "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"error\":{\"code\":\"session_turn_limit_exceeded\"}}}\n\n", "text/event-stream", 200},
	}
	for _, enabled := range []bool{false, true} {
		for _, credential := range []string{"", "Authorization", "X-Api-Key", "Api-Key"} {
			for _, tc := range cases {
				t.Run(fmt.Sprintf("%t/%s/%s", enabled, credential, tc.name), func(t *testing.T) {
					cfg := testConfig()
					cfg.responsesStateless = enabled
					cfg.upstreamKey = "fallback"
					var calls int
					handler := statelessTestMux(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						calls++
						if credential == "" && r.Header.Get("Authorization") != "Bearer fallback" {
							t.Error("fallback credential missing")
						}
						if credential != "" && r.Header.Get(credential) != "caller" {
							t.Error("caller credential changed")
						}
						if credential != "" && credential != "Authorization" && r.Header.Get("Authorization") != "" {
							t.Error("fallback overrode caller credential")
						}
						w.Header().Set("Content-Type", tc.contentType)
						w.Header().Set("X-Upstream", "unchanged")
						w.WriteHeader(tc.status)
						_, _ = io.WriteString(w, tc.body)
					}))
					request := httptest.NewRequest("POST", "/v1/responses", strings.NewReader(`{"input":"hi","previous_response_id":"old"}`))
					request.Header.Set("Idempotency-Key", "synthetic-no-retry")
					if credential != "" {
						request.Header.Set(credential, "caller")
					}
					recorder := httptest.NewRecorder()
					handler.ServeHTTP(recorder, request)
					if calls != 1 || recorder.Code != tc.status || recorder.Body.String() != tc.body || recorder.Header().Get("X-Upstream") != "unchanged" || recorder.Header().Get("Content-Type") != tc.contentType {
						t.Fatalf("calls=%d status=%d headers=%v body=%s", calls, recorder.Code, recorder.Header(), recorder.Body)
					}
				})
			}
		}
	}
}

func TestResponsesStatelessDisabledPreservesRawRequest(t *testing.T) {
	for _, method := range []string{"POST", "GET"} {
		t.Run(method, func(t *testing.T) {
			body := "not JSON\x00\xff"
			cfg := config{responsesStatelessHeaders: []string{"X-Session-Id"}}
			handler := statelessTestMux(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got, _ := io.ReadAll(r.Body)
				if string(got) != body || r.Header.Get("Content-Encoding") != "gzip" || r.Header.Get("X-Session-Id") != "keep" || r.URL.RawQuery != "q=x%2Fy" {
					t.Errorf("request changed: %q %v %s", got, r.Header, r.URL)
				}
				w.WriteHeader(204)
			}))
			request := httptest.NewRequest(method, "/v1/responses?q=x%2Fy", strings.NewReader(body))
			request.Header.Set("Content-Encoding", "gzip")
			request.Header.Set("X-Session-Id", "keep")
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != 204 {
				t.Fatalf("status=%d", recorder.Code)
			}
		})
	}
}

func TestResponsesStatelessOtherRoutesStillProxyRawBody(t *testing.T) {
	for _, target := range []struct{ method, path string }{
		{"GET", "/v1/responses"},
		{"POST", "/v1/responses/"},
		{"POST", "/v1/responses/resp_1"},
		{"POST", "/v1/chat/completions"},
	} {
		t.Run(target.method+" "+target.path, func(t *testing.T) {
			body := "not JSON\x00\xff"
			calls := 0
			cfg := config{responsesStateless: true, responsesStatelessHeaders: []string{"X-Session-Id"}}
			handler := statelessTestMux(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				got, _ := io.ReadAll(r.Body)
				if string(got) != body || r.URL.Path != target.path || r.Header.Get("X-Session-Id") != "keep" || r.Header.Get("Content-Encoding") != "gzip" {
					t.Errorf("non-target changed: %q %s %v", got, r.URL, r.Header)
				}
				w.WriteHeader(204)
			}))
			request := httptest.NewRequest(target.method, target.path, strings.NewReader(body))
			request.Header.Set("X-Session-Id", "keep")
			request.Header.Set("Content-Encoding", "gzip")
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != 204 || calls != 1 {
				t.Fatalf("status=%d calls=%d", recorder.Code, calls)
			}
		})
	}
}

func TestResponsesStatelessDoesNotChangeMessagesConversion(t *testing.T) {
	for _, api := range []bool{false, true} {
		t.Run(fmt.Sprintf("responses=%t", api), func(t *testing.T) {
			cfg := config{responsesAPI: api, responsesStateless: true, responsesStatelessHeaders: []string{"X-Session-Id"}}
			handler := statelessTestMux(t, cfg, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				want := "/v1/chat/completions"
				response := `{"id":"chat_1","choices":[]}`
				if api {
					want = "/v1/responses"
					response = `{"id":"resp_1","status":"completed","output":[]}`
				}
				if r.URL.Path != want {
					t.Errorf("path=%s, want=%s", r.URL.Path, want)
				}
				_, _ = io.WriteString(w, response)
			}))
			request := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
			// The native-only middleware must not impose these restrictions here.
			request.Header.Set("Content-Encoding", "gzip")
			request.Trailer = http.Header{"Digest": nil}
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, request)
			if recorder.Code != 200 {
				t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body)
			}
		})
	}
}

func TestResponsesStatelessStreamsWithoutWaitingForCompletion(t *testing.T) {
	const first = "event: response.created\ndata: {\"id\":\"resp_1\"}\n\n"
	const last = "event: response.completed\ndata: {\"id\":\"resp_1\"}\n\n"
	release := make(chan struct{}, 1)
	var calls atomic.Int32
	handler := statelessTestMux(t, config{responsesStateless: true}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, first)
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
			return
		}
		_, _ = io.WriteString(w, last)
	}))
	defer close(release)
	bridge := httptest.NewServer(handler)
	defer bridge.Close()
	client := &http.Client{Transport: &http.Transport{Proxy: nil}, Timeout: 5 * time.Second}
	defer client.CloseIdleConnections()
	response, err := client.Post(bridge.URL+"/v1/responses", "application/json", strings.NewReader(`{"input":"hi","previous_response_id":"old"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	gotFirst := make([]byte, len(first))
	if _, err := io.ReadFull(response.Body, gotFirst); err != nil || string(gotFirst) != first {
		t.Fatalf("first chunk before release=%q err=%v", gotFirst, err)
	}
	release <- struct{}{}
	gotLast, err := io.ReadAll(response.Body)
	if err != nil || string(gotLast) != last || calls.Load() != 1 {
		t.Fatalf("last=%q err=%v calls=%d", gotLast, err, calls.Load())
	}
}
