package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

// Anthropic Messages wire types. Optional request scalars stay pointers so an
// absent field is never turned into an explicit zero.

type claudeRequest struct {
	Model         string          `json:"model"`
	MaxTokens     *int            `json:"max_tokens,omitempty"`
	System        json.RawMessage `json:"system,omitempty"`
	Messages      []claudeMessage `json:"messages"`
	Tools         []claudeTool    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage `json:"tool_choice,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	OutputConfig  *claudeOutput   `json:"output_config,omitempty"`
}

type claudeOutput struct {
	Effort string `json:"effort,omitempty"`
}

type claudeMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type claudeBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	Source    *claudeSource   `json:"source,omitempty"`
}

type claudeSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type claudeTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type claudeResponse struct {
	ID         string           `json:"id"`
	Type       string           `json:"type"`
	Role       string           `json:"role"`
	Model      string           `json:"model"`
	Content    []claudeOutBlock `json:"content"`
	StopReason string           `json:"stop_reason"`
	StopSeq    *string          `json:"stop_sequence"`
	Usage      claudeUsage      `json:"usage"`
}

type claudeOutBlock struct {
	Type      string `json:"type"`
	Text      string `json:"text,omitempty"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	ID        string `json:"id,omitempty"`
	Name      string `json:"name,omitempty"`
	Input     any    `json:"input,omitempty"`
}

// thinkingSignature stands in for the signature Anthropic attaches to thinking
// blocks. Clients replay it in later turns, and the conversion back to Chat
// Completions drops thinking blocks, so a stable placeholder is enough.
const thinkingSignature = "messages-bridge"

type claudeUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// convertClaudeToOpenAI maps a Messages request onto a Chat Completions request
// that carries only fields the OpenAI schema defines.
func convertClaudeToOpenAI(req *claudeRequest, cfg config) (*openAIRequest, error) {
	out := &openAIRequest{
		Model:       mapModel(req.Model, cfg),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}
	if req.Stream {
		out.StreamOptions = &openAIStreamOptions{IncludeUsage: true}
	}
	switch len(req.StopSequences) {
	case 0:
	case 1:
		out.Stop = req.StopSequences[0]
	default:
		out.Stop = req.StopSequences
	}
	if cfg.reasoningEffort && req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		out.ReasoningEffort = req.OutputConfig.Effort
	}

	if system := systemText(req.System); system != "" {
		out.Messages = append(out.Messages, openAIMessage{Role: "system", Content: system})
	}

	for _, message := range req.Messages {
		converted, err := convertClaudeMessage(message)
		if err != nil {
			return nil, err
		}
		out.Messages = append(out.Messages, converted...)
	}

	for _, tool := range req.Tools {
		if tool.Name == "" {
			continue
		}
		out.Tools = append(out.Tools, openAITool{
			Type: "function",
			Function: openAIFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  cleanInputSchema(tool.InputSchema),
			},
		})
	}
	out.ToolChoice = convertToolChoice(req.ToolChoice)

	if len(out.Messages) == 0 {
		return nil, fmt.Errorf("request has no messages after conversion")
	}
	return out, nil
}

func convertClaudeMessage(message claudeMessage) ([]openAIMessage, error) {
	text, blocks, isString, err := parseContent(message.Content)
	if err != nil {
		return nil, err
	}

	switch message.Role {
	case "user":
		if isString {
			return []openAIMessage{{Role: "user", Content: text}}, nil
		}
		return convertClaudeUserBlocks(blocks), nil
	case "assistant":
		if isString {
			return []openAIMessage{{Role: "assistant", Content: text}}, nil
		}
		return convertClaudeAssistantBlocks(blocks), nil
	case "system":
		// Claude clients may place a system block inside messages; keep the role
		// so the upstream sees the same instruction ordering.
		if isString {
			return []openAIMessage{{Role: "system", Content: text}}, nil
		}
		texts := make([]string, 0, len(blocks))
		for _, block := range blocks {
			if block.Type == "text" {
				texts = append(texts, block.Text)
			}
		}
		if len(texts) == 0 {
			return nil, nil
		}
		return []openAIMessage{{Role: "system", Content: strings.Join(texts, "\n")}}, nil
	default:
		return nil, fmt.Errorf("unsupported message role %q", message.Role)
	}
}

func convertClaudeUserBlocks(blocks []claudeBlock) []openAIMessage {
	var out []openAIMessage
	var parts []any
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, map[string]any{"type": "text", "text": block.Text})
		case "image":
			if url := imageURL(block.Source); url != "" {
				parts = append(parts, map[string]any{"type": "image_url", "image_url": map[string]any{"url": url}})
			}
		case "tool_result":
			out = append(out, openAIMessage{
				Role:       "tool",
				ToolCallID: block.ToolUseID,
				Content:    toolResultText(block.Content),
			})
		}
	}
	if len(parts) == 0 {
		return out
	}
	if text, ok := joinTextParts(parts); ok {
		return append(out, openAIMessage{Role: "user", Content: text})
	}
	return append(out, openAIMessage{Role: "user", Content: parts})
}

func convertClaudeAssistantBlocks(blocks []claudeBlock) []openAIMessage {
	message := openAIMessage{Role: "assistant"}
	texts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			texts = append(texts, block.Text)
		case "tool_use":
			arguments := "{}"
			if len(block.Input) > 0 && string(block.Input) != "null" {
				arguments = string(block.Input)
			}
			message.ToolCalls = append(message.ToolCalls, openAIToolCall{
				ID:       block.ID,
				Type:     "function",
				Function: openAIToolCallFunction{Name: block.Name, Arguments: arguments},
			})
		}
	}
	if len(texts) > 0 {
		message.Content = strings.Join(texts, "\n")
	}
	if message.Content == nil && len(message.ToolCalls) == 0 {
		return nil
	}
	return []openAIMessage{message}
}

func parseContent(raw json.RawMessage) (string, []claudeBlock, bool, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return "", nil, true, nil
	}
	if strings.HasPrefix(trimmed, "\"") {
		var text string
		if err := json.Unmarshal(raw, &text); err != nil {
			return "", nil, false, fmt.Errorf("decode message content: %w", err)
		}
		return text, nil, true, nil
	}
	var blocks []claudeBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", nil, false, fmt.Errorf("decode message content blocks: %w", err)
	}
	return "", blocks, false, nil
}

func systemText(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	if strings.HasPrefix(trimmed, "\"") {
		var text string
		if err := json.Unmarshal(raw, &text); err == nil {
			return text
		}
		return ""
	}
	var blocks []claudeBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var builder strings.Builder
	for _, block := range blocks {
		if block.Type == "text" {
			builder.WriteString(block.Text)
		}
	}
	return builder.String()
}

func toolResultText(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	if strings.HasPrefix(trimmed, "\"") {
		var text string
		if err := json.Unmarshal(raw, &text); err == nil {
			return text
		}
		return ""
	}
	var blocks []claudeBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	texts := make([]string, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			texts = append(texts, block.Text)
		case "image":
			texts = append(texts, "[image omitted]")
		}
	}
	return strings.Join(texts, "\n")
}

func imageURL(source *claudeSource) string {
	if source == nil {
		return ""
	}
	if source.Type == "url" && source.URL != "" {
		return source.URL
	}
	if source.Data == "" {
		return ""
	}
	mediaType := source.MediaType
	if mediaType == "" {
		mediaType = "image/png"
	}
	return "data:" + mediaType + ";base64," + source.Data
}

func joinTextParts(parts []any) (string, bool) {
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		entry, ok := part.(map[string]any)
		if !ok || entry["type"] != "text" {
			return "", false
		}
		text, _ := entry["text"].(string)
		texts = append(texts, text)
	}
	return strings.Join(texts, "\n"), true
}

// cleanInputSchema turns an Anthropic input_schema into the JSON-Schema subset
// upstreams actually accept.
func cleanInputSchema(raw json.RawMessage) any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return json.RawMessage(raw)
	}
	sanitizeSchema(schema)
	if len(schema) == 0 {
		return nil
	}
	return schema
}

// sanitizeSchema trims a tool schema to the subset upstreams accept. Both
// removals are load-bearing: an empty "enum" and an unsupported "pattern" each
// fail the entire request with a bare "Invalid request".
func sanitizeSchema(value any) {
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if key == "$schema" {
				delete(typed, key)
				continue
			}
			if key == "enum" {
				if values, ok := child.([]any); ok && len(values) == 0 {
					delete(typed, key)
					continue
				}
			}
			if key == "pattern" {
				if pattern, ok := child.(string); !ok || !keepPattern(pattern) {
					delete(typed, key)
				}
				continue
			}
			sanitizeSchema(child)
		}
	case []any:
		for _, child := range typed {
			sanitizeSchema(child)
		}
	}
}

// keepPattern reports whether a JSON-Schema pattern survives the upstream
// validator. Its engine has no lookaround, no backreferences, and no Unicode
// classes, and any pattern it cannot compile rejects the whole request, so
// patterns outside that subset are dropped rather than risk the call.
func keepPattern(pattern string) bool {
	if strings.Contains(pattern, "(?") || strings.Contains(pattern, `\p`) || strings.Contains(pattern, `\P`) {
		return false
	}
	for digit := '1'; digit <= '9'; digit++ {
		if strings.Contains(pattern, `\`+string(digit)) {
			return false
		}
	}
	_, err := regexp.Compile(pattern)
	return err == nil
}

func convertToolChoice(raw json.RawMessage) any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var choice struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &choice); err != nil {
		return nil
	}
	switch choice.Type {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "none":
		return "none"
	case "tool":
		if choice.Name == "" {
			return "auto"
		}
		return map[string]any{"type": "function", "function": map[string]string{"name": choice.Name}}
	default:
		return nil
	}
}

func mapModel(model string, cfg config) string {
	if mapped, ok := cfg.modelMap[model]; ok && mapped != "" {
		return mapped
	}
	return model
}
