# LLM Log Viewer

A pure frontend web application for viewing and exploring LLM request/response logs from the NovelRAG agent system.

## Features

- **Directory Browsing**: Browse and select from multiple YAML files in a directory
- **Single File Selection**: Load individual YAML log files directly in the browser
- **Left Panel Directory View**: Navigate files with a dedicated directory panel showing only YAML files
- **Hierarchical View**: Navigate through pursuits → LLM calls → request/response details
- **Smart Request Formatting**: System prompts are prominently displayed with clear section headers, user queries are secondary, and technical parameters are shown as reference
- **Easy Copy/Paste**: Click any content area to select all text, double-click to copy to clipboard
- **Collapsible Interface**: 
  - Pursuits are collapsed by default, showing goal and metadata
  - LLM calls within pursuits are collapsed by default, showing template name and timing
  - Requests are collapsed by default (since they're usually long prompts)
  - Responses are expanded by default for easy reading
- **Statistics**: View summary statistics including pursuit count, total LLM calls, and unique templates used
- **Responsive Design**: Works on desktop and mobile devices
- **No Backend Required**: Pure HTML/CSS/JavaScript - works offline

## Usage

1. **Open the Viewer**: 
   - Open `index.html` in any modern web browser

2. **Load Log Files**:
   - **Directory Mode**: Click "Select Directory" to browse multiple YAML files in a folder
   - **Single File Mode**: Click "Select File" to load one YAML file

3. **Browse Files**:
   - Use the left directory panel to see all available YAML files
   - Click on any file in the directory panel to view its contents
   - Selected file is highlighted in the directory panel

4. **Navigate the Log**:
   - Click on pursuit headers to expand/collapse pursuit details
   - Click on LLM call headers to expand/collapse call details  
   - Click on "Request" or "Response" headers to toggle their visibility
   - **Click any content area to select all text** for easy copying
   - **Double-click any content area to copy to clipboard**
   - Use the statistics at the top to get an overview

## Log File Format

The viewer expects YAML files with the following structure:

```yaml
pursuits:
  - goal: "Create a character named John"
    started_at: "2025-10-20T10:30:00"
    completed_at: "2025-10-20T10:35:00"
    llm_calls:
      - template_name: "character_creation.jinja2"
        timestamp: "2025-10-20T10:30:15"
        duration_ms: 1250
        request:
          messages:
            - role: "system"
              content: "You are a creative writing assistant..."
            - role: "user"
              content: "Please create a character."
          response_format: "json_object"
        response:
          content: "{\\"name\\": \\"John Smith\\", \\"age\\": 35...}"
```

## Files

- `index.html` - Main HTML structure and layout
- `styles.css` - All styling and responsive design
- `script.js` - JavaScript functionality for parsing and displaying logs
- `README.md` - This documentation

## Browser Compatibility

- Chrome/Chromium 80+
- Firefox 75+
- Safari 13+
- Edge 80+

Requires support for:
- ES6+ JavaScript features
- CSS Grid and Flexbox
- File API for local file reading

## Development

To extend or modify the viewer:

1. **Adding New Log Fields**: Update the `formatRequest()` and `formatResponse()` methods in `script.js`
2. **Styling Changes**: Modify `styles.css` - uses CSS custom properties for easy theming
3. **New Features**: Add to the `LLMLogViewer` class in `script.js`

## Troubleshooting

**File won't load**: 
- Ensure the file is valid YAML format
- Check browser console for specific error messages

**Display Issues**: 
- Try refreshing the page
- Ensure JavaScript is enabled
- Check that all files are in the same directory

**Performance with Large Files**:
- Very large log files (>10MB) may load slowly
- Consider splitting large logs into smaller files