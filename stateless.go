package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

func parseResponsesStatelessHeaders(raw string) ([]string, error) {
	var headers []string
	seen := make(map[string]bool)
	for _, name := range strings.Split(raw, ",") {
		name = http.CanonicalHeaderKey(strings.TrimSpace(name))
		if name == "" {
			continue
		}
		switch name {
		case "Session-Id", "X-Session-Id", "Conversation-Id", "X-Conversation-Id":
		default:
			return nil, fmt.Errorf("RESPONSES_STATELESS_DROP_HEADERS only supports Session-Id, X-Session-Id, Conversation-Id, X-Conversation-Id")
		}
		if !seen[name] {
			headers = append(headers, name)
			seen[name] = true
		}
	}
	return headers, nil
}

// withResponsesStateless only changes native Responses requests. In particular,
// the Messages converter and all responses still use their existing paths.
func withResponsesStateless(next http.Handler, cfg config) http.Handler {
	if !cfg.responsesStateless {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/responses" {
			next.ServeHTTP(w, r)
			return
		}
		for _, encoding := range r.Header.Values("Content-Encoding") {
			if encoding = strings.TrimSpace(encoding); encoding != "" && !strings.EqualFold(encoding, "identity") {
				writeClaudeError(w, http.StatusUnsupportedMediaType, "invalid_request_error", "stateless Responses mode does not support encoded request bodies")
				return
			}
		}
		if len(r.Trailer) != 0 || len(r.Header.Values("Trailer")) != 0 {
			writeClaudeError(w, http.StatusUnsupportedMediaType, "invalid_request_error", "stateless Responses mode does not support request trailers")
			return
		}

		var body []byte
		if r.Body != nil {
			var err error
			body, err = io.ReadAll(io.LimitReader(r.Body, maxRequestBytes+1))
			_ = r.Body.Close()
			if len(body) > maxRequestBytes {
				writeClaudeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "request body exceeds stateless Responses mode's 64 MiB limit")
				return
			}
			if err != nil {
				writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
				return
			}
		}
		// Some transports only expose undeclared trailers after EOF.
		if len(r.Trailer) != 0 {
			writeClaudeError(w, http.StatusUnsupportedMediaType, "invalid_request_error", "stateless Responses mode does not support request trailers")
			return
		}
		fields, err := decodeResponsesObject(body)
		if err != nil {
			writeClaudeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}

		_, previous := fields["previous_response_id"]
		_, conversation := fields["conversation"]
		if previous || conversation {
			// Rewriting a signed body would invalidate its integrity information.
			for _, name := range []string{"Content-MD5", "Digest", "Content-Digest", "Repr-Digest", "Signature", "Signature-Input"} {
				if len(r.Header.Values(name)) != 0 {
					writeClaudeError(w, http.StatusUnsupportedMediaType, "invalid_request_error", "stateless Responses mode cannot rewrite a body with digest or signature headers")
					return
				}
			}
			delete(fields, "previous_response_id")
			delete(fields, "conversation")
			body, err = json.Marshal(fields)
			if err != nil {
				writeClaudeError(w, http.StatusInternalServerError, "api_error", "could not encode upstream request")
				return
			}
		}

		forward := r.Clone(r.Context())
		forward.Body = io.NopCloser(bytes.NewReader(body))
		forward.ContentLength = int64(len(body))
		forward.Header.Del("Content-Length")
		forward.Header.Del("Transfer-Encoding")
		forward.TransferEncoding = nil
		// Do not make a buffered POST newly replayable by the transport.
		forward.GetBody = nil
		for _, name := range cfg.responsesStatelessHeaders {
			forward.Header.Del(name)
		}
		next.ServeHTTP(w, forward)
	})
}

// RawMessage preserves values (including large numbers and unknown fields)
// without recursively interpreting input history or tool arguments.
func decodeResponsesObject(body []byte) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(body))
	token, err := decoder.Token()
	if err != nil || token != json.Delim('{') {
		return nil, fmt.Errorf("stateless Responses mode requires a single JSON object")
	}
	fields := make(map[string]json.RawMessage)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return nil, fmt.Errorf("invalid JSON body")
		}
		name, ok := token.(string)
		if !ok {
			return nil, fmt.Errorf("invalid JSON object key")
		}
		if _, exists := fields[name]; exists {
			return nil, fmt.Errorf("stateless Responses mode does not support duplicate top-level JSON keys")
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil, fmt.Errorf("invalid JSON body")
		}
		fields[name] = value
	}
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("invalid JSON body")
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); err != io.EOF {
		return nil, fmt.Errorf("stateless Responses mode requires a single JSON object")
	}
	return fields, nil
}
