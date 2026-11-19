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

func handleMessagesResponses(w http.ResponseWriter, r *http.Request, cfg config) {
	body, err := io.ReadAll(io.LimitReader(r.Body, maxRequestBytes))
	if err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}
	var request claudeRequest
	if err = json.Unmarshal(body, &request); err != nil {
		writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON body: "+err.Error())
		return
	}
	converted, err := convertClaudeToResponses(&request, cfg)
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
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.upstreamBase.String()+"/v1/responses", bytes.NewReader(payload))
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
	if cfg.transport != nil {
		client.Transport = cfg.transport
	}
	response, err := client.Do(httpRequest)
	if err != nil {
		log.Printf("upstream request failed: %v", err)
		writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream request failed")
		return
	}
	defer response.Body.Close()
	if response.StatusCode >= http.StatusBadRequest {
		errorBody, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		log.Printf("upstream rejected request (status=%d): %s", response.StatusCode, errorBody)
		writeClaudeError(w, response.StatusCode, claudeErrorType(response.StatusCode), upstreamMessage(errorBody))
		return
	}
	if request.Stream {
		if strings.Contains(response.Header.Get("Content-Type"), "text/event-stream") {
			if err := translateResponsesStream(w, response.Body, request.Model); err != nil {
				log.Printf("responses stream translation ended: %v", err)
			}
			return
		}
		var completion responsesOutput
		if err := json.NewDecoder(response.Body).Decode(&completion); err != nil {
			log.Printf("decode non-stream Responses body: %v", err)
			writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream returned an unreadable body")
			return
		}
		if err := writeSyntheticStream(w, convertResponsesToClaude(&completion, request.Model)); err != nil {
			log.Printf("synthetic stream failed: %v", err)
		}
		return
	}
	var completion responsesOutput
	if err = json.NewDecoder(response.Body).Decode(&completion); err != nil {
		log.Printf("decode responses upstream body: %v", err)
		writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream returned an unreadable body")
		return
	}
	writeJSON(w, http.StatusOK, convertResponsesToClaude(&completion, request.Model))
}

// readResponsesStream collects the final Responses object from its SSE lifecycle.
// Anthropic's event stream is emitted as a synthetic stream after upstream completion.
func readResponsesStream(body io.Reader) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(body, maxRequestBytes))
	if err != nil {
		return nil, err
	}
	var final []byte
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			continue
		}
		var event struct {
			Type     string          `json:"type"`
			Response json.RawMessage `json:"response"`
		}
		if json.Unmarshal([]byte(payload), &event) == nil && (event.Type == "response.completed" || event.Type == "response.incomplete") {
			final = event.Response
		}
	}
	if len(final) == 0 {
		return nil, io.ErrUnexpectedEOF
	}
	return final, nil
}
