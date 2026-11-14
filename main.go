/*
Copyright (C) 2023-2026 xingguangcuican6666

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
*/

// Command messages-bridge sits between an API gateway and an OpenAI-compatible
// upstream. Anthropic Messages requests are converted to Chat Completions on the
// way out and the answer is converted back, so a Claude client can talk to an
// upstream that only speaks OpenAI. Every other path is proxied unchanged.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// version is stamped by the container build so the running binary can be
// identified from the first log line.
var version = "dev"

type config struct {
	addr            string
	upstreamBase    *url.URL
	upstreamKey     string
	modelMap        map[string]string
	reasoningEffort bool
	timeout         time.Duration
	logBodies       bool
	// transport is shared by the converter and the passthrough proxy so both
	// honour the same proxy configuration.
	transport *http.Transport
}

func loadConfig() (config, error) {
	cfg := config{
		addr:            envOr("LISTEN_ADDR", ":8080"),
		upstreamKey:     os.Getenv("UPSTREAM_API_KEY"),
		reasoningEffort: envOr("PASS_REASONING_EFFORT", "true") != "false",
		timeout:         10 * time.Minute,
		logBodies:       envOr("LOG_BODIES", "false") == "true",
	}

	rawBase := strings.TrimSpace(os.Getenv("UPSTREAM_BASE_URL"))
	if rawBase == "" {
		return cfg, fmt.Errorf("UPSTREAM_BASE_URL is required")
	}
	base, err := url.Parse(rawBase)
	if err != nil {
		return cfg, fmt.Errorf("UPSTREAM_BASE_URL is invalid: %w", err)
	}
	if base.Scheme == "" || base.Host == "" {
		return cfg, fmt.Errorf("UPSTREAM_BASE_URL must be absolute, got %q", rawBase)
	}
	base.Path = strings.TrimSuffix(base.Path, "/")
	cfg.upstreamBase = base

	// Clone the default transport so HTTP_PROXY / HTTPS_PROXY / NO_PROXY keep
	// working, then let UPSTREAM_PROXY point at an explicit proxy. A bare
	// host:port is accepted and read as http, so 192.168.31.109:7890 works.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if raw := strings.TrimSpace(os.Getenv("UPSTREAM_PROXY")); raw != "" {
		if !strings.Contains(raw, "://") {
			// Only host:port gets the implicit scheme, so a plain word stays an
			// error instead of silently becoming a proxy hostname.
			if !strings.Contains(raw, ":") {
				return cfg, fmt.Errorf("UPSTREAM_PROXY must be a proxy URL or host:port, got %q", raw)
			}
			raw = "http://" + raw
		}
		proxyURL, err := url.Parse(raw)
		if err != nil || proxyURL.Scheme == "" || proxyURL.Host == "" {
			return cfg, fmt.Errorf("UPSTREAM_PROXY must be a proxy URL or host:port, got %q", raw)
		}
		transport.Proxy = http.ProxyURL(proxyURL)
	}
	cfg.transport = transport

	if raw := strings.TrimSpace(os.Getenv("UPSTREAM_TIMEOUT")); raw != "" {
		parsed, err := time.ParseDuration(raw)
		if err != nil {
			return cfg, fmt.Errorf("UPSTREAM_TIMEOUT is invalid: %w", err)
		}
		cfg.timeout = parsed
	}

	if raw := strings.TrimSpace(os.Getenv("MODEL_MAP")); raw != "" {
		if err := json.Unmarshal([]byte(raw), &cfg.modelMap); err != nil {
			return cfg, fmt.Errorf("MODEL_MAP must be a JSON object of client model to upstream model: %w", err)
		}
	}
	return cfg, nil
}

func describeProxy(cfg config) string {
	if cfg.transport == nil || cfg.transport.Proxy == nil {
		return "none"
	}
	request, _ := http.NewRequest(http.MethodGet, cfg.upstreamBase.String(), nil)
	proxyURL, err := cfg.transport.Proxy(request)
	if err != nil || proxyURL == nil {
		return "none"
	}
	return proxyURL.String()
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		log.Fatalf("config: %v", err)
	}

	proxy, err := newPassthroughProxy(cfg)
	if err != nil {
		log.Fatalf("proxy: %v", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/messages", func(w http.ResponseWriter, r *http.Request) {
		handleMessages(w, r, cfg)
	})
	mux.HandleFunc("POST /v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		handleCountTokens(w, r)
	})
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "version": version})
	})
	mux.Handle("/", proxy)

	server := &http.Server{
		Addr:              cfg.addr,
		Handler:           logRequests(mux),
		ReadHeaderTimeout: 30 * time.Second,
	}
	log.Printf("messages-bridge %s listening on %s, upstream %s, proxy %s", version, cfg.addr, cfg.upstreamBase, describeProxy(cfg))
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("serve: %v", err)
	}
}

func logRequests(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		recorder := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(recorder, r)
		log.Printf("%s %s -> %d (%s)", r.Method, r.URL.RequestURI(), recorder.status, time.Since(start).Round(time.Millisecond))
	})
}

type statusRecorder struct {
	http.ResponseWriter
	status int
}

func (r *statusRecorder) WriteHeader(code int) {
	r.status = code
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Flush() {
	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeClaudeError(w http.ResponseWriter, status int, errType, message string) {
	writeJSON(w, status, map[string]any{
		"type":  "error",
		"error": map[string]string{"type": errType, "message": message},
	})
}

func estimateTokens(body []byte) int {
	// Rough but stable: providers count a token per few bytes of JSON text.
	tokens := len(body) / 4
	if tokens < 1 {
		tokens = 1
	}
	return tokens
}
