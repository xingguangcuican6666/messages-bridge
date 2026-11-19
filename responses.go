package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// responsesRequest is the subset of the OpenAI Responses API used by this bridge.
type responsesRequest struct {
	Model           string            `json:"model"`
	Input           []any             `json:"input"`
	Instructions    string            `json:"instructions,omitempty"`
	MaxOutputTokens *int              `json:"max_output_tokens,omitempty"`
	Temperature     *float64          `json:"temperature,omitempty"`
	TopP            *float64          `json:"top_p,omitempty"`
	Stream          bool              `json:"stream,omitempty"`
	Tools           []responsesTool   `json:"tools,omitempty"`
	ToolChoice      any               `json:"tool_choice,omitempty"`
	Reasoning       map[string]string `json:"reasoning,omitempty"`
}

type responsesTool struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type responsesOutput struct {
	ID     string `json:"id"`
	Model  string `json:"model"`
	Status string `json:"status"`
	Output []struct {
		Type      string                        `json:"type"`
		Role      string                        `json:"role"`
		Content   []struct{ Type, Text string } `json:"content"`
		ID        string                        `json:"id"`
		CallID    string                        `json:"call_id"`
		Name      string                        `json:"name"`
		Arguments string                        `json:"arguments"`
		Summary   []struct{ Type, Text string } `json:"summary"`
	} `json:"output"`
	Usage *struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
}

func convertClaudeToResponses(req *claudeRequest, cfg config) (*responsesRequest, error) {
	out := &responsesRequest{Model: mapModel(req.Model, cfg), Stream: req.Stream, Temperature: req.Temperature, TopP: req.TopP, MaxOutputTokens: req.MaxTokens}
	out.Instructions = systemText(req.System)
	if cfg.reasoningEffort && req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		out.Reasoning = map[string]string{"effort": req.OutputConfig.Effort}
	}
	for _, msg := range req.Messages {
		text, blocks, isString, err := parseContent(msg.Content)
		if err != nil {
			return nil, err
		}
		if isString {
			out.Input = append(out.Input, map[string]any{"role": msg.Role, "content": text})
			continue
		}
		var content []any
		for _, b := range blocks {
			switch b.Type {
			case "text":
				typ := "input_text"
				if msg.Role == "assistant" {
					typ = "output_text"
				}
				content = append(content, map[string]any{"type": typ, "text": b.Text})
			case "image":
				if u := imageURL(b.Source); u != "" {
					content = append(content, map[string]any{"type": "input_image", "image_url": u})
				}
			case "tool_result":
				out.Input = append(out.Input, map[string]any{"type": "function_call_output", "call_id": b.ToolUseID, "output": toolResultText(b.Content)})
			case "tool_use":
				args := b.Input
				if len(args) == 0 {
					args = json.RawMessage(`{}`)
				}
				out.Input = append(out.Input, map[string]any{"type": "function_call", "call_id": b.ID, "name": b.Name, "arguments": string(args)})
			}
		}
		if len(content) > 0 {
			out.Input = append(out.Input, map[string]any{"role": msg.Role, "content": content})
		}
	}
	for _, t := range req.Tools {
		if t.Name != "" {
			out.Tools = append(out.Tools, responsesTool{Type: "function", Name: t.Name, Description: t.Description, Parameters: cleanInputSchema(t.InputSchema)})
		}
	}
	out.ToolChoice = convertResponsesToolChoice(req.ToolChoice)
	if len(out.Input) == 0 && out.Instructions == "" {
		return nil, fmt.Errorf("request has no input after conversion")
	}
	return out, nil
}

func convertResponsesToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var choice struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &choice) != nil {
		return nil
	}
	switch choice.Type {
	case "auto":
		return "auto"
	case "none":
		return "none"
	case "any":
		return "required"
	case "tool":
		return map[string]any{"type": "function", "name": choice.Name}
	default:
		return nil
	}
}

func convertResponsesToClaude(resp *responsesOutput, clientModel string) *claudeResponse {
	out := &claudeResponse{ID: messageID(resp.ID), Type: "message", Role: "assistant", Model: clientModel, Content: []claudeOutBlock{}, StopReason: "end_turn", Usage: claudeUsage{}}
	if resp.Usage != nil {
		out.Usage.InputTokens = resp.Usage.InputTokens
		out.Usage.OutputTokens = resp.Usage.OutputTokens
	}
	var reasoning strings.Builder
	for _, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if strings.Contains(c.Type, "reasoning") {
					reasoning.WriteString(c.Text)
				} else if c.Type == "output_text" || c.Type == "text" {
					out.Content = append(out.Content, claudeOutBlock{Type: "text", Text: c.Text})
				}
			}
			for _, c := range item.Summary {
				reasoning.WriteString(c.Text)
			}
		case "reasoning":
			for _, c := range item.Summary {
				reasoning.WriteString(c.Text)
			}
		case "function_call":
			out.Content = append(out.Content, claudeOutBlock{Type: "tool_use", ID: item.CallID, Name: item.Name, Input: toolInput(item.Arguments)})
		}
	}
	if reasoning.Len() > 0 {
		out.Content = append([]claudeOutBlock{{Type: "thinking", Thinking: reasoning.String(), Signature: thinkingSignature}}, out.Content...)
	}
	if len(out.Content) > 0 && out.Content[len(out.Content)-1].Type == "tool_use" {
		out.StopReason = "tool_use"
	}
	if resp.Status == "incomplete" && resp.IncompleteDetails != nil && resp.IncompleteDetails.Reason == "max_output_tokens" {
		out.StopReason = "max_tokens"
	}
	return out
}
