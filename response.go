package main

import (
	"encoding/json"
	"strings"
)

// convertOpenAIToClaude maps a Chat Completions response onto a Messages
// response, echoing the model name the client asked for.
func convertOpenAIToClaude(resp *openAIResponse, clientModel string) *claudeResponse {
	out := &claudeResponse{
		ID:         messageID(resp.ID),
		Type:       "message",
		Role:       "assistant",
		Model:      clientModel,
		Content:    []claudeOutBlock{},
		StopReason: "end_turn",
		Usage:      claudeUsage{},
	}
	if resp.Usage != nil {
		out.Usage.InputTokens = resp.Usage.PromptTokens
		out.Usage.OutputTokens = resp.Usage.CompletionTokens
	}
	if len(resp.Choices) == 0 {
		return out
	}

	choice := resp.Choices[0]
	// Anthropic puts thinking ahead of the answer, and clients render it as the
	// collapsible chain of thought.
	if reasoning := choice.Message.ReasoningContent; reasoning != "" {
		out.Content = append(out.Content, claudeOutBlock{Type: "thinking", Thinking: reasoning, Signature: thinkingSignature})
	}
	if text := contentText(choice.Message.Content); text != "" {
		out.Content = append(out.Content, claudeOutBlock{Type: "text", Text: text})
	}
	for _, call := range choice.Message.ToolCalls {
		out.Content = append(out.Content, claudeOutBlock{
			Type:  "tool_use",
			ID:    call.ID,
			Name:  call.Function.Name,
			Input: toolInput(call.Function.Arguments),
		})
	}
	out.StopReason = stopReason(choice.FinishReason, len(choice.Message.ToolCalls) > 0)
	return out
}

func contentText(content any) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []any:
		var builder strings.Builder
		for _, part := range typed {
			entry, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if text, ok := entry["text"].(string); ok {
				builder.WriteString(text)
			}
		}
		return builder.String()
	default:
		return ""
	}
}

func toolInput(arguments string) any {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return map[string]any{}
	}
	var input any
	if err := json.Unmarshal([]byte(trimmed), &input); err != nil {
		return map[string]any{"_raw": arguments}
	}
	if input == nil {
		return map[string]any{}
	}
	return input
}

func stopReason(finishReason string, hasTools bool) string {
	switch finishReason {
	case "tool_calls", "function_call":
		return "tool_use"
	case "length":
		return "max_tokens"
	case "content_filter":
		return "end_turn"
	case "":
		if hasTools {
			return "tool_use"
		}
		return "end_turn"
	default:
		if hasTools {
			return "tool_use"
		}
		return "end_turn"
	}
}

func messageID(upstreamID string) string {
	trimmed := strings.TrimSpace(upstreamID)
	if trimmed == "" {
		return "msg_bridge"
	}
	if strings.HasPrefix(trimmed, "msg_") {
		return trimmed
	}
	return "msg_" + trimmed
}
