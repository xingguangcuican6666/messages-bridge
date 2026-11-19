package main

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

type responsesStreamItem struct {
	ID     string `json:"id"`
	CallID string `json:"call_id"`
	Name   string `json:"name"`
	Type   string `json:"type"`
}

func translateResponsesStream(w http.ResponseWriter, body io.Reader, clientModel string) error {
	sse, ok := newSSEWriter(w)
	if !ok {
		return io.ErrUnexpectedEOF
	}
	t := newStreamTranslator(sse, clientModel)
	items := map[int]responsesStreamItem{}
	reader := bufio.NewReaderSize(body, 64*1024)
	var dataLines []string
	finished := false
	handle := func(payload string) bool {
		payload = strings.TrimSpace(payload)
		if payload == "" {
			return true
		}
		var e struct {
			Type     string `json:"type"`
			Response struct {
				ID    string `json:"id"`
				Usage struct {
					Input  int `json:"input_tokens"`
					Output int `json:"output_tokens"`
				} `json:"usage"`
			} `json:"response"`
			Item        responsesStreamItem `json:"item"`
			OutputIndex int                 `json:"output_index"`
			Delta       string              `json:"delta"`
		}
		if json.Unmarshal([]byte(payload), &e) != nil {
			return true
		}
		switch e.Type {
		case "response.created", "response.in_progress":
			t.start(e.Response.ID)
		case "response.output_item.added":
			items[e.OutputIndex] = e.Item
			t.start(e.Response.ID)
			if e.Item.Type == "function_call" {
				id := e.Item.CallID
				if id == "" {
					id = e.Item.ID
				}
				idx := e.OutputIndex
				t.toolDelta(e.Response.ID, openAIToolCall{Index: &idx, ID: id, Function: openAIToolCallFunction{Name: e.Item.Name}})
			}
		case "response.reasoning_summary_text.delta":
			t.thinkingDelta(e.Response.ID, e.Delta)
		case "response.output_text.delta":
			t.textDelta(e.Response.ID, e.Delta)
		case "response.function_call_arguments.delta":
			item := items[e.OutputIndex]
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			idx := e.OutputIndex
			t.toolDelta(e.Response.ID, openAIToolCall{Index: &idx, ID: id, Function: openAIToolCallFunction{Name: item.Name, Arguments: e.Delta}})
		case "response.completed", "response.incomplete":
			t.start(e.Response.ID)
			t.usage = openAIUsage{PromptTokens: e.Response.Usage.Input, CompletionTokens: e.Response.Usage.Output}
			t.haveUsage = true
			if e.Type == "response.incomplete" {
				t.finishReason = "length"
			}
			t.finish(e.Response.ID)
			finished = true
		case "response.failed", "error":
			if !t.started {
				writeClaudeError(w, http.StatusBadGateway, "api_error", "upstream Responses request failed")
			}
			return false
		}
		return true
	}
	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				if len(dataLines) > 0 {
					if !handle(strings.Join(dataLines, "\n")) {
						return nil
					}
					dataLines = dataLines[:0]
				}
			} else if rest, ok := strings.CutPrefix(trimmed, "data:"); ok {
				dataLines = append(dataLines, strings.TrimSpace(rest))
			}
		}
		if err != nil {
			if len(dataLines) > 0 {
				handle(strings.Join(dataLines, "\n"))
			}
			if err != io.EOF {
				return err
			}
			break
		}
	}
	if !finished && t.started {
		t.finish("")
	}
	return nil
}
