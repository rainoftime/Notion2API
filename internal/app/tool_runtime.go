package app

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

const (
	localToolListFiles  = "local_list_files"
	localToolReadFile   = "local_read_file"
	localToolSearchText = "local_search_text"
	maxToolIterations   = 8
	maxToolResultBytes  = 64 * 1024
	maxToolReadBytes    = 32 * 1024
	maxToolSearchHits   = 50
)

type functionToolDefinition struct {
	Name        string
	Description string
	Parameters  map[string]any
}

type toolLoopContext struct {
	Definitions []functionToolDefinition
	Workspace   string
	Choice      string
}

type toolLoopEvent struct {
	CallID     string
	Name       string
	Arguments  string
	ResultJSON string
}

type toolCallEnvelope struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type toolCallResult struct {
	Calls        []toolLoopEvent
	HasToolCalls bool
	VisibleText  string
}

type structuredToolCallPayload struct {
	ToolCalls []struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
	Content string `json:"content"`
}

func parseFunctionToolsAny(raw any) []functionToolDefinition {
	items := sliceValue(raw)
	if len(items) == 0 {
		return nil
	}
	out := make([]functionToolDefinition, 0, len(items))
	seen := map[string]struct{}{}
	for _, item := range items {
		tool := mapValue(item)
		if len(tool) == 0 {
			continue
		}
		if strings.TrimSpace(stringValue(tool["type"])) != "function" {
			continue
		}
		fn := mapValue(tool["function"])
		name := strings.TrimSpace(stringValue(fn["name"]))
		if name == "" {
			continue
		}
		if _, ok := seen[name]; ok {
			continue
		}
		seen[name] = struct{}{}
		out = append(out, functionToolDefinition{
			Name:        name,
			Description: strings.TrimSpace(stringValue(fn["description"])),
			Parameters:  mapValue(fn["parameters"]),
		})
	}
	if len(out) == 0 {
		return nil
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Name < out[j].Name
	})
	return out
}

func supportedLocalToolDefinitions(definitions []functionToolDefinition) []functionToolDefinition {
	if len(definitions) == 0 {
		return nil
	}
	out := make([]functionToolDefinition, 0, len(definitions))
	for _, item := range definitions {
		switch strings.TrimSpace(item.Name) {
		case localToolListFiles, localToolReadFile, localToolSearchText:
			out = append(out, item)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func newToolLoopContext(definitions []functionToolDefinition, toolChoice any) (*toolLoopContext, error) {
	definitions = supportedLocalToolDefinitions(definitions)
	if len(definitions) == 0 {
		return nil, nil
	}
	workspace, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	choice := normalizeToolChoice(toolChoice)
	if choice == "none" {
		return nil, nil
	}
	return &toolLoopContext{Definitions: definitions, Workspace: filepath.Clean(workspace), Choice: choice}, nil
}

func normalizeToolChoice(raw any) string {
	switch value := raw.(type) {
	case string:
		clean := strings.TrimSpace(strings.ToLower(value))
		switch clean {
		case "", "auto":
			return "auto"
		case "none", "required":
			return clean
		}
	case map[string]any:
		if strings.TrimSpace(strings.ToLower(stringValue(value["type"]))) == "function" {
			return "required"
		}
	}
	return "auto"
}

func (a *App) runPromptWithLocalTools(ctx context.Context, request PromptRunRequest, tools *toolLoopContext) (InferenceResult, error) {
	if tools == nil || len(tools.Definitions) == 0 {
		return a.runPrompt(nil, request)
	}
	working := request
	basePrompt := request.Prompt
	history := make([]toolLoopEvent, 0, 4)
	for iter := 0; iter < maxToolIterations; iter++ {
		working.Prompt = buildToolLoopPrompt(basePrompt, request.HiddenPrompt, tools.Definitions, history)
		working.HiddenPrompt = ""
		result, err := a.runPrompt(nil, working)
		if err != nil {
			return InferenceResult{}, err
		}
		parsed := parseToolCallResponse(result.Text)
		if !parsed.HasToolCalls {
			result.Text = firstNonEmpty(strings.TrimSpace(parsed.VisibleText), strings.TrimSpace(result.Text))
			result.Prompt = request.Prompt
			if len(history) > 0 {
				result.ToolCalls = make([]InferenceToolCall, 0, len(history))
				for _, item := range history {
					result.ToolCalls = append(result.ToolCalls, InferenceToolCall{
						ID:         item.CallID,
						Type:       "function",
						Name:       item.Name,
						Arguments:  item.Arguments,
						ResultJSON: item.ResultJSON,
					})
				}
			}
			return result, nil
		}
		for _, call := range parsed.Calls {
			output, execErr := executeLocalToolCall(ctx, tools.Workspace, call.Name, call.Arguments)
			if execErr != nil {
				output = map[string]any{"ok": false, "error": execErr.Error()}
			}
			body, marshalErr := json.Marshal(output)
			if marshalErr != nil {
				body = []byte(`{"ok":false,"error":"failed to encode tool result"}`)
			}
			call.ResultJSON = truncateBytesString(string(body), maxToolResultBytes)
			history = append(history, call)
		}
	}
	return InferenceResult{}, fmt.Errorf("tool loop exceeded %d iterations", maxToolIterations)
}

func buildToolLoopPrompt(basePrompt string, hiddenPrompt string, definitions []functionToolDefinition, history []toolLoopEvent) string {
	parts := []string{
		"You may use local workspace tools through this bridge before answering.",
		"When you need a tool, prefer replying with JSON like {\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"tool_name\",\"arguments\":\"{...}\"}}]}",
		"If needed for compatibility, you may also reply with one or more <tool_call>{...}</tool_call> blocks.",
		"When you have enough information, answer normally with no <tool_call> tags.",
		"Stay within the available tools and prefer targeted reads over large files.",
	}
	if strings.TrimSpace(hiddenPrompt) != "" {
		parts = append(parts, formatPromptSection("hidden_context", hiddenPrompt))
	}
	toolLines := make([]string, 0, len(definitions))
	for _, item := range definitions {
		line := item.Name
		if item.Description != "" {
			line += ": " + item.Description
		}
		if len(item.Parameters) > 0 {
			if body, err := json.Marshal(item.Parameters); err == nil {
				line += "\nparameters=" + string(body)
			}
		}
		toolLines = append(toolLines, line)
	}
	parts = append(parts, formatPromptSection("available_tools", strings.Join(toolLines, "\n\n")))
	if len(history) > 0 {
		historyLines := make([]string, 0, len(history)*2)
		for _, item := range history {
			historyLines = append(historyLines,
				fmt.Sprintf("assistant tool request (%s): %s", item.Name, collapseWhitespace(item.Arguments)),
				fmt.Sprintf("tool result (%s): %s", item.Name, item.ResultJSON),
			)
		}
		parts = append(parts, formatPromptSection("tool_history", strings.Join(historyLines, "\n")))
	}
	parts = append(parts, formatPromptSection("user_request", basePrompt))
	return strings.Join(parts, "\n\n")
}

func parseToolCallResponse(text string) toolCallResult {
	clean := strings.TrimSpace(text)
	result := toolCallResult{VisibleText: clean}
	if clean == "" {
		return result
	}
	if structured, ok := parseStructuredToolCallResponse(clean); ok {
		return structured
	}
	const openTag = "<tool_call>"
	const closeTag = "</tool_call>"
	visibleParts := make([]string, 0, 1)
	for {
		start := strings.Index(clean, openTag)
		if start < 0 {
			if strings.TrimSpace(clean) != "" {
				visibleParts = append(visibleParts, strings.TrimSpace(clean))
			}
			break
		}
		if strings.TrimSpace(clean[:start]) != "" {
			visibleParts = append(visibleParts, strings.TrimSpace(clean[:start]))
		}
		clean = clean[start+len(openTag):]
		end := strings.Index(clean, closeTag)
		if end < 0 {
			visibleParts = append(visibleParts, strings.TrimSpace(openTag+clean))
			break
		}
		block := strings.TrimSpace(clean[:end])
		clean = clean[end+len(closeTag):]
		if block == "" {
			continue
		}
		var payload toolCallEnvelope
		if err := json.Unmarshal([]byte(block), &payload); err != nil {
			visibleParts = append(visibleParts, strings.TrimSpace(openTag+block+closeTag))
			continue
		}
		argumentsBody, _ := json.Marshal(payload.Arguments)
		result.Calls = append(result.Calls, toolLoopEvent{
			CallID:    randomUUID(),
			Name:      strings.TrimSpace(payload.Name),
			Arguments: string(argumentsBody),
		})
	}
	result.VisibleText = strings.TrimSpace(strings.Join(visibleParts, "\n\n"))
	result.HasToolCalls = len(result.Calls) > 0
	return result
}

func parseStructuredToolCallResponse(text string) (toolCallResult, bool) {
	var payload structuredToolCallPayload
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		return toolCallResult{}, false
	}
	result := toolCallResult{VisibleText: strings.TrimSpace(payload.Content)}
	for _, item := range payload.ToolCalls {
		if strings.TrimSpace(item.Type) != "function" {
			continue
		}
		name := strings.TrimSpace(item.Function.Name)
		args := strings.TrimSpace(item.Function.Arguments)
		if name == "" {
			continue
		}
		if args == "" {
			args = "{}"
		}
		callID := strings.TrimSpace(item.ID)
		if callID == "" {
			callID = randomUUID()
		}
		result.Calls = append(result.Calls, toolLoopEvent{
			CallID:    callID,
			Name:      name,
			Arguments: args,
		})
	}
	result.HasToolCalls = len(result.Calls) > 0
	return result, result.HasToolCalls || result.VisibleText != ""
}

func executeLocalToolCall(ctx context.Context, workspace string, name string, argumentsJSON string) (map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	args := map[string]any{}
	if strings.TrimSpace(argumentsJSON) != "" {
		if err := json.Unmarshal([]byte(argumentsJSON), &args); err != nil {
			return nil, fmt.Errorf("invalid tool arguments: %w", err)
		}
	}
	switch strings.TrimSpace(name) {
	case localToolListFiles:
		return executeLocalListFiles(workspace, args)
	case localToolReadFile:
		return executeLocalReadFile(workspace, args)
	case localToolSearchText:
		return executeLocalSearchText(ctx, workspace, args)
	default:
		return nil, fmt.Errorf("unsupported tool: %s", name)
	}
}

func resolveWorkspacePath(workspace string, raw string) (string, string, error) {
	clean := strings.TrimSpace(raw)
	if clean == "" || clean == "." {
		return workspace, ".", nil
	}
	var abs string
	if filepath.IsAbs(clean) {
		abs = filepath.Clean(clean)
	} else {
		abs = filepath.Clean(filepath.Join(workspace, clean))
	}
	rel, err := filepath.Rel(workspace, abs)
	if err != nil {
		return "", "", err
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", "", fmt.Errorf("path escapes workspace: %s", raw)
	}
	if rel == "." {
		return abs, rel, nil
	}
	return abs, filepath.ToSlash(rel), nil
}

func executeLocalListFiles(workspace string, args map[string]any) (map[string]any, error) {
	abs, rel, err := resolveWorkspacePath(workspace, stringValue(args["path"]))
	if err != nil {
		return nil, err
	}
	maxDepth := intValue(args["max_depth"], 3)
	if maxDepth < 0 {
		maxDepth = 0
	}
	if maxDepth > 6 {
		maxDepth = 6
	}
	entries := make([]map[string]any, 0, 64)
	err = filepath.WalkDir(abs, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == abs {
			return nil
		}
		currentRel, relErr := filepath.Rel(abs, path)
		if relErr != nil {
			return relErr
		}
		depth := strings.Count(filepath.ToSlash(currentRel), "/") + 1
		if depth > maxDepth {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		info, infoErr := d.Info()
		if infoErr != nil {
			return infoErr
		}
		entries = append(entries, map[string]any{
			"path":   filepath.ToSlash(currentRel),
			"is_dir": d.IsDir(),
			"size":   info.Size(),
		})
		if len(entries) >= 200 {
			return filepath.SkipAll
		}
		return nil
	})
	if err != nil && err != filepath.SkipAll {
		return nil, err
	}
	return map[string]any{
		"ok":             true,
		"workspace_path": rel,
		"max_depth":      maxDepth,
		"entries":        entries,
	}, nil
}

func executeLocalReadFile(workspace string, args map[string]any) (map[string]any, error) {
	abs, rel, err := resolveWorkspacePath(workspace, stringValue(args["path"]))
	if err != nil {
		return nil, err
	}
	body, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	if len(body) > maxToolReadBytes {
		body = body[:maxToolReadBytes]
	}
	text := string(body)
	lines := strings.Split(text, "\n")
	startLine := intValue(args["start_line"], 1)
	endLine := intValue(args["end_line"], len(lines))
	if startLine < 1 {
		startLine = 1
	}
	if endLine < startLine {
		endLine = startLine
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}
	selected := ""
	if len(lines) > 0 && startLine <= len(lines) {
		selected = strings.Join(lines[startLine-1:endLine], "\n")
	}
	return map[string]any{
		"ok":         true,
		"path":       rel,
		"start_line": startLine,
		"end_line":   endLine,
		"content":    truncateBytesString(selected, maxToolResultBytes),
	}, nil
}

func executeLocalSearchText(ctx context.Context, workspace string, args map[string]any) (map[string]any, error) {
	query := strings.TrimSpace(stringValue(args["query"]))
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	rootArg := firstNonEmptyString(args["path"], ".")
	abs, rel, err := resolveWorkspacePath(workspace, rootArg)
	if err != nil {
		return nil, err
	}
	pattern := strings.TrimSpace(stringValue(args["glob"]))
	cmd := []string{"rg", "-n", "--no-heading", "--color", "never", query, abs}
	if pattern != "" {
		cmd = []string{"rg", "-n", "--no-heading", "--color", "never", "-g", pattern, query, abs}
	}
	out, err := runLocalSearchCommand(ctx, cmd)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	results := make([]map[string]any, 0, minInt(len(lines), maxToolSearchHits))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 3)
		if len(parts) < 3 {
			continue
		}
		pathRel, relErr := filepath.Rel(workspace, parts[0])
		if relErr != nil {
			pathRel = parts[0]
		}
		results = append(results, map[string]any{
			"path": filepath.ToSlash(pathRel),
			"line": parts[1],
			"text": truncateBytesString(parts[2], 500),
		})
		if len(results) >= maxToolSearchHits {
			break
		}
	}
	return map[string]any{
		"ok":      true,
		"path":    rel,
		"query":   query,
		"glob":    pattern,
		"matches": results,
	}, nil
}

var runLocalSearchCommand = func(ctx context.Context, argv []string) (string, error) {
	if len(argv) == 0 {
		return "", fmt.Errorf("missing command")
	}
	parsed := make([]string, 0, len(argv))
	for _, part := range argv {
		if strings.TrimSpace(part) == "" {
			continue
		}
		parsed = append(parsed, part)
	}
	if len(parsed) == 0 {
		return "", fmt.Errorf("missing command")
	}
	cmd := exec.CommandContext(ctx, parsed[0], parsed[1:]...)
	body, err := cmd.CombinedOutput()
	if err != nil {
		clean := strings.TrimSpace(string(body))
		if clean == "" {
			return "", err
		}
		return "", fmt.Errorf("%v: %s", err, clean)
	}
	return string(body), nil
}

func intValue(raw any, fallback int) int {
	switch value := raw.(type) {
	case int:
		return value
	case int32:
		return int(value)
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		parsed, err := value.Int64()
		if err == nil {
			return int(parsed)
		}
	case string:
		if parsed := strings.TrimSpace(value); parsed != "" {
			var n int
			if _, err := fmt.Sscanf(parsed, "%d", &n); err == nil {
				return n
			}
		}
	}
	return fallback
}

func truncateBytesString(text string, limit int) string {
	if limit <= 0 || len(text) <= limit {
		return text
	}
	return text[:limit]
}
