# Trace Viewer

A pure frontend web application for viewing and exploring hierarchical trace spans produced by the NovelRAG tracer framework.

## Features

- **Hierarchical Span Tree**: Navigate the full trace tree — Session → Intent → Pursuit → Tool Call → LLM Call
- **Color-Coded Span Kinds**: Each span level has a distinct colour badge for quick visual scanning
- **Directory Browsing**: Browse and select from multiple YAML trace files in a directory
- **Single File Selection**: Load individual trace files directly in the browser
- **LLM Call Details**: Expand LLM call spans to see request messages (system prompt, user query), response content, and token usage
- **Error Highlighting**: Spans with errors are highlighted in red with the error message shown inline
- **Collapsible Interface**:
  - All spans are collapsed by default — click to reveal children
  - Inside LLM calls, requests are collapsed and responses are expanded by default
- **Statistics**: Total spans, LLM call count, total tokens, and error count
- **Easy Copy/Paste**: Double-click any message or response block to copy to clipboard
- **Responsive Design**: Works on desktop and mobile devices
- **No Backend Required**: Pure HTML/CSS/JavaScript — works offline

## Usage

1. **Open the Viewer**:
   - Open `index.html` in any modern web browser

2. **Load Trace Files**:
   - **Directory Mode**: Click "Select Directory" to browse multiple YAML files in a folder
   - **Single File Mode**: Click "Select File" to load one trace file

3. **Navigate the Trace**:
   - Click on any span header to expand/collapse its children
   - LLM call spans show Request and Response sections inside
   - Click "Request" or "Response" headers to toggle their visibility
   - **Double-click any content area to copy to clipboard**
   - Use the statistics bar at the top for a quick overview

## Trace File Format

The viewer expects YAML files matching the `Span.to_dict()` output from the NovelRAG tracer:

```yaml
kind: session
name: shell_session
span_id: a1b2c3d4e5f6
start_time: '2026-02-15T14:30:00.000000'
end_time: '2026-02-15T14:35:42.123456'
duration_ms: 342123.46
status: ok
children:
- kind: intent
  name: handle_request
  span_id: b2c3d4e5f6a7
  ...
  children:
  - kind: pursuit
    name: handle_goal
    attributes:
      goal: Create a fantasy character
    children:
    - kind: llm_call
      name: content_generation
      attributes:
        model: gpt-4o
        request:
          messages:
          - role: system
            content: "You are a creative writing assistant..."
          - role: human
            content: "Create a character."
        response:
          content: '{"name": "Elena Brightforge", ...}'
        token_usage:
          prompt_tokens: 245
          completion_tokens: 389
          total_tokens: 634
```

## Files

- `index.html` — Main HTML structure and layout
- `styles.css` — All styling, span-kind colours, and responsive design
- `script.js` — JavaScript `TraceViewer` class for parsing and rendering span trees
- `sample_log.yaml` — Example trace file for testing
- `README.md` — This documentation

## Browser Compatibility

- Chrome/Chromium 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Troubleshooting

**File won't load**:
- Ensure the root object has a `kind` field (e.g. `kind: session`)
- Check browser console for specific error messages

**Display Issues**:
- Try refreshing the page
- Ensure JavaScript is enabled

**Performance with Large Files**:
- Very large trace files (>10MB) may load slowly
- Consider splitting traces by session