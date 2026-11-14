package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
)

const maxRequestBytes = 64 << 20

func handleMessages(w http.ResponseWriter, r *http.Request, cfg config) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBytes))
	if err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}

	var request claudeRequest
	if err := json.Unmarshal(body, &request); err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}

	converted, err := convertClaudeToOpenAI(&request, cfg)
	if err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	payload, err := json.Marshal(converted)
	if err != nil {
		writeClaudeError(w, http.StatusInternalServerError, "api_error", "could not encode upstream request")
		return
	}
	if cfg.logBodies {
		log.Printf("converted upstream body: %s", payload)
	}

	ctx := r.Context()
	if !request.Stream {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, cfg.timeout)
		defer cancel()
	}

	upstreamURL := cfg.upstreamBase.String() + "/v1/chat/completions"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(payload))
	if err != nil {
		writeClaudeError(w, http.StatusInternalServerError, "api_error", "could not build upstream request")
		return
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	if key := resolveUpstreamKey(r, cfg); key != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+key)
	}
	if request.Stream {
		httpRequest.Header.Set("Accept", "text/event-stream")
	} else {
		httpRequest.Header.Set("Accept", "application/json")
	}

	client := &http.Client{}
	response, err := client.Do(httpRequest)
	if err != nil {
		log.Printf("upstream request failed: %v", err)
		writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream request failed")
		return
	}
	defer func() {
		_ = response.Body.Close()
	}()

	if response.StatusCode >= http.StatusBadRequest {
		errorBody, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		log.Printf("upstream rejected request (status=%d): %s", response.StatusCode, errorBody)
		writeClaudeError(w, response.StatusCode, claudeErrorType(response.StatusCode), upstreamMessage(errorBody))
		return
	}

	if request.Stream {
		if strings.Contains(response.Header.Get("Content-Type"), "text/event-stream") {
			if err := translateStream(w, response.Body, request.Model); err != nil {
				log.Printf("stream translation ended: %v", err)
			}
			return
		}
		// The upstream answered in one piece even though streaming was asked for.
		var completion openAIResponse
		if err := json.NewDecoder(response.Body).Decode(&completion); err != nil {
			log.Printf("decode non-stream upstream body: %v", err)
			writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream returned an unreadable body")
			return
		}
		if err := writeSyntheticStream(w, convertOpenAIToClaude(&completion, request.Model)); err != nil {
			log.Printf("synthetic stream failed: %v", err)
		}
		return
	}

	var completion openAIResponse
	if err := json.NewDecoder(response.Body).Decode(&completion); err != nil {
		log.Printf("decode upstream body: %v", err)
		writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream returned an unreadable body")
		return
	}
	writeJSON(w, http.StatusOK, convertOpenAIToClaude(&completion, request.Model))
}

func handleCountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBytes))
	if err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}
	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": estimateTokens(body)})
}

// resolveUpstreamKey prefers the configured upstream credential and otherwise
// forwards whatever the calling gateway used.
func resolveUpstreamKey(r *http.Request, cfg config) string {
	if cfg.upstreamKey != "" {
		return cfg.upstreamKey
	}
	if auth := strings.TrimSpace(r.Header.Get("Authorization")); auth != "" {
		if token, ok := strings.CutPrefix(auth, "Bearer "); ok {
			return strings.TrimSpace(token)
		}
		return auth
	}
	if key := strings.TrimSpace(r.Header.Get("x-api-key")); key != "" {
		return key
	}
	return strings.TrimSpace(r.Header.Get("api-key"))
}

func claudeErrorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

// upstreamMessage extracts the human-readable part of an upstream error body.
func upstreamMessage(body []byte) string {
	text := strings.TrimSpace(string(body))
	if text == "" {
		return "upstream rejected the request"
	}
	var nested struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &nested); err == nil {
		if nested.Error.Message != "" {
			return nested.Error.Message
		}
		if nested.Message != "" {
			return nested.Message
		}
	}
	var flat struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal(body, &flat); err == nil && flat.Error != "" {
		return flat.Error
	}
	if len(text) > 500 {
		return text[:500]
	}
	return text
}
