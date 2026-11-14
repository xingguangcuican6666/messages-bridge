package main

import (
	"bufio"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
)

// sseWriter emits Anthropic server-sent events and flushes every one of them, so
// the client sees deltas as the upstream produces them.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

func newSSEWriter(w http.ResponseWriter) (*sseWriter, bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	return &sseWriter{w: w, flusher: flusher}, true
}

func (s *sseWriter) emit(event string, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		log.Printf("stream: marshal %s event: %v", event, err)
		return
	}
	var builder strings.Builder
	builder.WriteString("event: ")
	builder.WriteString(event)
	builder.WriteString("\ndata: ")
	builder.Write(data)
	builder.WriteString("\n\n")
	if _, err := io.WriteString(s.w, builder.String()); err != nil {
		return
	}
	s.flusher.Flush()
}

type toolStreamState struct {
	id       string
	name     string
	blockIdx int
	started  bool
}

// streamTranslator turns Chat Completions deltas into the Messages event stream.
type streamTranslator struct {
	sse          *sseWriter
	clientMdl    string
	started      bool
	nextBlock    int
	textIdx      int
	textOpen     bool
	tools        map[int]*toolStreamState
	order        []int
	usage        openAIUsage
	haveUsage    bool
	finishReason string
}

func newStreamTranslator(sse *sseWriter, clientModel string) *streamTranslator {
	return &streamTranslator{sse: sse, clientMdl: clientModel, textIdx: -1, tools: map[int]*toolStreamState{}}
}

func (t *streamTranslator) start(id string) {
	if t.started {
		return
	}
	t.started = true
	t.sse.emit("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            messageID(id),
			"type":          "message",
			"role":          "assistant",
			"model":         t.clientMdl,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         map[string]int{"input_tokens": 0, "output_tokens": 0},
		},
	})
}

func (t *streamTranslator) closeText() {
	if !t.textOpen {
		return
	}
	t.sse.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": t.textIdx})
	t.textOpen = false
}

func (t *streamTranslator) textDelta(id, text string) {
	if text == "" {
		return
	}
	t.start(id)
	if !t.textOpen {
		t.textIdx = t.nextBlock
		t.nextBlock++
		t.sse.emit("content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         t.textIdx,
			"content_block": map[string]any{"type": "text", "text": ""},
		})
		t.textOpen = true
	}
	t.sse.emit("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": t.textIdx,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

func (t *streamTranslator) toolDelta(id string, call openAIToolCall) {
	t.start(id)
	index := 0
	if call.Index != nil {
		index = *call.Index
	}
	state, ok := t.tools[index]
	if !ok {
		state = &toolStreamState{blockIdx: -1}
		t.tools[index] = state
		t.order = append(t.order, index)
	}
	if call.ID != "" {
		state.id = call.ID
	}
	if call.Function.Name != "" {
		state.name = call.Function.Name
	}
	arguments := call.Function.Arguments
	if !state.started {
		if state.name == "" && arguments == "" {
			return
		}
		t.closeText()
		state.blockIdx = t.nextBlock
		t.nextBlock++
		state.started = true
		t.sse.emit("content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": state.blockIdx,
			"content_block": map[string]any{
				"type":  "tool_use",
				"id":    state.id,
				"name":  state.name,
				"input": map[string]any{},
			},
		})
	}
	if arguments == "" {
		return
	}
	t.sse.emit("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": state.blockIdx,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": arguments},
	})
}

func (t *streamTranslator) noteUsage(usage *openAIUsage) {
	if usage == nil {
		return
	}
	t.usage = *usage
	t.haveUsage = true
}

func (t *streamTranslator) finish(id string) {
	t.start(id)
	t.closeText()
	for _, index := range t.order {
		state := t.tools[index]
		if state.started {
			t.sse.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": state.blockIdx})
			state.started = false
		}
	}
	usage := map[string]int{"input_tokens": t.usage.PromptTokens, "output_tokens": t.usage.CompletionTokens}
	t.sse.emit("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason(t.finishReason, len(t.tools) > 0), "stop_sequence": nil},
		"usage": usage,
	})
	t.sse.emit("message_stop", map[string]any{"type": "message_stop"})
}

// translateStream consumes the upstream SSE body and writes Anthropic events.
func translateStream(w http.ResponseWriter, body io.Reader, clientModel string) error {
	sse, ok := newSSEWriter(w)
	if !ok {
		return io.ErrUnexpectedEOF
	}
	translator := newStreamTranslator(sse, clientModel)

	reader := bufio.NewReaderSize(body, 64*1024)
	var dataLines []string
	upstreamID := ""

	handlePayload := func(payload string) bool {
		payload = strings.TrimSpace(payload)
		if payload == "" {
			return true
		}
		if payload == "[DONE]" {
			return false
		}
		var chunk openAIResponse
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			log.Printf("stream: skip unparsable chunk: %v", err)
			return true
		}
		if chunk.ID != "" {
			upstreamID = chunk.ID
		}
		translator.noteUsage(chunk.Usage)
		for _, choice := range chunk.Choices {
			if choice.FinishReason != "" {
				translator.finishReason = choice.FinishReason
			}
			if text := contentText(choice.Delta.Content); text != "" {
				translator.textDelta(upstreamID, text)
			}
			for _, call := range choice.Delta.ToolCalls {
				translator.toolDelta(upstreamID, call)
			}
		}
		return true
	}

	for {
		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				if len(dataLines) > 0 {
					keepGoing := handlePayload(strings.Join(dataLines, "\n"))
					dataLines = dataLines[:0]
					if !keepGoing {
						break
					}
				}
			} else if rest, ok := strings.CutPrefix(trimmed, "data:"); ok {
				dataLines = append(dataLines, strings.TrimSpace(rest))
			}
		}
		if err != nil {
			if len(dataLines) > 0 {
				handlePayload(strings.Join(dataLines, "\n"))
			}
			if err != io.EOF {
				return err
			}
			break
		}
	}

	translator.finish(upstreamID)
	return nil
}

// writeSyntheticStream replays a complete response as an Anthropic event stream,
// which keeps streaming clients working when the upstream answers in one piece.
func writeSyntheticStream(w http.ResponseWriter, resp *claudeResponse) error {
	sse, ok := newSSEWriter(w)
	if !ok {
		return io.ErrUnexpectedEOF
	}
	translator := newStreamTranslator(sse, resp.Model)
	translator.noteUsage(&openAIUsage{PromptTokens: resp.Usage.InputTokens, CompletionTokens: resp.Usage.OutputTokens})
	translator.start(resp.ID)
	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			translator.textDelta(resp.ID, block.Text)
		case "tool_use":
			translator.closeText()
			blockIdx := translator.nextBlock
			translator.nextBlock++
			sse.emit("content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": blockIdx,
				"content_block": map[string]any{
					"type":  "tool_use",
					"id":    block.ID,
					"name":  block.Name,
					"input": map[string]any{},
				},
			})
			encoded, _ := json.Marshal(block.Input)
			sse.emit("content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": blockIdx,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": string(encoded)},
			})
			sse.emit("content_block_stop", map[string]any{"type": "content_block_stop", "index": blockIdx})
		}
	}
	translator.closeText()
	sse.emit("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": resp.StopReason, "stop_sequence": nil},
		"usage": map[string]int{"input_tokens": resp.Usage.InputTokens, "output_tokens": resp.Usage.OutputTokens},
	})
	sse.emit("message_stop", map[string]any{"type": "message_stop"})
	return nil
}
