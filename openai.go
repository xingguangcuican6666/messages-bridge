package main

// OpenAI Chat Completions wire types. Only the fields this bridge emits or reads
// are modelled; everything else in an upstream response is ignored.

type openAIRequest struct {
	Model           string               `json:"model"`
	Messages        []openAIMessage      `json:"messages"`
	MaxTokens       *int                 `json:"max_tokens,omitempty"`
	Temperature     *float64             `json:"temperature,omitempty"`
	TopP            *float64             `json:"top_p,omitempty"`
	Stop            any                  `json:"stop,omitempty"`
	Stream          bool                 `json:"stream,omitempty"`
	StreamOptions   *openAIStreamOptions `json:"stream_options,omitempty"`
	Tools           []openAITool         `json:"tools,omitempty"`
	ToolChoice      any                  `json:"tool_choice,omitempty"`
	ReasoningEffort string               `json:"reasoning_effort,omitempty"`
}

type openAIStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    any              `json:"content,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

type openAIToolCall struct {
	Index    *int                   `json:"index,omitempty"`
	ID       string                 `json:"id,omitempty"`
	Type     string                 `json:"type,omitempty"`
	Function openAIToolCallFunction `json:"function"`
}

type openAIToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type openAITool struct {
	Type     string         `json:"type"`
	Function openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type openAIResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message      openAIResponseMessage `json:"message"`
		Delta        openAIResponseMessage `json:"delta"`
		FinishReason string                `json:"finish_reason"`
	} `json:"choices"`
	Usage *openAIUsage `json:"usage,omitempty"`
}

type openAIResponseMessage struct {
	Role      string           `json:"role"`
	Content   any              `json:"content"`
	ToolCalls []openAIToolCall `json:"tool_calls,omitempty"`
}
