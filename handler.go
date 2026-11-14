package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
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
	if !request.Stream && cfg.timeout > 0 {
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

	// A nil *http.Transport stored in the field is not a nil interface, so only
	// set it when the config actually built one.
	client := &http.Client{}
	if cfg.transport != nil {
		client.Transport = cfg.transport
	}
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

// resolveUpstreamKey forwards the credential the calling gateway attached, so
// the key lives in new-api's channel settings instead of this service. The
// configured key is only a fallback for callers that send no credential.
func resolveUpstreamKey(r *http.Request, cfg config) string {
	if key := callerCredential(r); key != "" {
		return key
	}
	return cfg.upstreamKey
}

// callerCredential returns the credential the calling gateway attached, if any.
func callerCredential(r *http.Request) string {
	for _, header := range []string{"Authorization", "x-api-key", "api-key"} {
		if key := credentialFromHeader(r.Header.Get(header)); key != "" {
			return key
		}
	}
	return ""
}

func credentialFromHeader(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	if token, ok := strings.CutPrefix(trimmed, "Bearer "); ok {
		return strings.TrimSpace(token)
	}
	return trimmed
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
		Details struct {
			Type   string `json:"type"`
			Limit  int    `json:"limit"`
			Actual int    `json:"actual"`
		} `json:"details"`
	}
	if err := json.Unmarshal(body, &nested); err == nil {
		// A bare "Invalid request" tells the client nothing, so surface the
		// reason the upstream attached. Context overflow is the one a client
		// can act on: it has to compact before retrying.
		if nested.Details.Type == "context_length" {
			return fmt.Sprintf("prompt is too long: %d tokens > %d maximum", nested.Details.Actual, nested.Details.Limit)
		}
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
