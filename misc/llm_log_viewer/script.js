/**
 * Trace Viewer – JavaScript functionality
 * Renders hierarchical span trees (SESSION → INTENT → PURSUIT → TOOL_CALL → LLM_CALL)
 * produced by the NovelRAG tracer framework.
 */

class TraceViewer {
    constructor() {
        this.currentTrace = null;
        this.currentDirectory = null;
        this.selectedFile = null;
        this.files = new Map();

        this.initializeElements();
        this.bindEvents();
    }

    // ------------------------------------------------------------------
    // Initialization
    // ------------------------------------------------------------------

    initializeElements() {
        this.fileInput = document.getElementById('fileInput');
        this.singleFileInput = document.getElementById('singleFileInput');
        this.fileName = document.getElementById('fileName');

        this.directoryPath = document.getElementById('directoryPath');
        this.directoryContent = document.getElementById('directoryContent');

        this.stats = document.getElementById('stats');
        this.totalSpans = document.getElementById('totalSpans');
        this.llmCallCount = document.getElementById('llmCallCount');
        this.totalTokens = document.getElementById('totalTokens');
        this.errorCount = document.getElementById('errorCount');

        this.logContent = document.getElementById('logContent');
    }

    bindEvents() {
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) this.loadDirectory(e.target.files);
        });
        this.singleFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) this.loadSingleFile(e.target.files[0]);
        });
    }

    // ------------------------------------------------------------------
    // File / directory handling  (unchanged logic, cleaned up)
    // ------------------------------------------------------------------

    loadDirectory(fileList) {
        this.files.clear();
        const yamlFiles = Array.from(fileList).filter(f =>
            f.name.toLowerCase().endsWith('.yaml') || f.name.toLowerCase().endsWith('.yml')
        );
        if (yamlFiles.length === 0) {
            this.fileName.textContent = 'No YAML files found in directory';
            this.showDirectoryEmpty('No YAML files found');
            return;
        }
        yamlFiles.forEach(f => this.files.set(f.name, f));
        const rel = yamlFiles[0].webkitRelativePath || yamlFiles[0].name;
        const dir = rel.includes('/') ? rel.substring(0, rel.lastIndexOf('/')) : 'Selected Directory';
        this.currentDirectory = dir;
        this.directoryPath.textContent = dir;
        this.fileName.textContent = `${yamlFiles.length} YAML files found`;
        this.displayDirectoryContents(yamlFiles);
    }

    loadSingleFile(file) {
        this.files.clear();
        this.files.set(file.name, file);
        this.currentDirectory = 'Single File';
        this.directoryPath.textContent = 'Single File';
        this.fileName.textContent = file.name;
        this.displayDirectoryContents([file]);
        this.selectFile(file.name);
    }

    displayDirectoryContents(files) {
        files.sort((a, b) => a.name.localeCompare(b.name));
        this.directoryContent.innerHTML = files.map(f => `
            <div class="file-item" data-filename="${f.name}">
                <span class="file-icon">📄</span>
                <span class="file-name-text">${f.name}</span>
                <span class="file-size">${this.formatFileSize(f.size)}</span>
            </div>
        `).join('');
        this.directoryContent.querySelectorAll('.file-item').forEach(item => {
            item.addEventListener('click', () => this.selectFile(item.dataset.filename));
        });
    }

    showDirectoryEmpty(message = 'No directory selected') {
        this.directoryContent.innerHTML = `
            <div class="directory-empty">
                <p>${message}</p>
                <p class="help-text">Use "Select Directory" button to browse trace files</p>
            </div>`;
    }

    async selectFile(filename) {
        this.directoryContent.querySelectorAll('.file-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.filename === filename);
        });
        this.selectedFile = filename;
        const file = this.files.get(filename);
        if (!file) return;
        try {
            const content = await this.readFileContent(file);
            this.parseAndDisplay(content, filename);
        } catch (error) {
            this.showError(`Error reading file: ${error.message}`);
        }
    }

    readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    // ------------------------------------------------------------------
    // Parse & render
    // ------------------------------------------------------------------

    parseAndDisplay(content, filename) {
        try {
            const data = jsyaml.load(content);
            if (!data || !data.kind) {
                throw new Error('Invalid trace format: missing "kind" field on root span');
            }
            this.currentTrace = data;
            this.updateStats(data);
            this.displayTrace(data);
        } catch (error) {
            this.showError(`Error parsing ${filename}: ${error.message}`);
        }
    }

    // ------------------------------------------------------------------
    // Stats
    // ------------------------------------------------------------------

    /** Recursively walk the tree to compute aggregate stats. */
    updateStats(root) {
        let spans = 0, llmCalls = 0, tokens = 0, errors = 0;
        const walk = (span) => {
            spans++;
            if (span.kind === 'llm_call') {
                llmCalls++;
                const tu = span.attributes?.token_usage;
                if (tu?.total_tokens) tokens += tu.total_tokens;
            }
            if (span.status === 'error') errors++;
            (span.children || []).forEach(walk);
        };
        walk(root);

        this.totalSpans.textContent = spans;
        this.llmCallCount.textContent = llmCalls;
        this.totalTokens.textContent = tokens.toLocaleString();
        this.errorCount.textContent = errors;
        this.errorCount.parentElement.classList.toggle('has-errors', errors > 0);
        this.stats.style.display = 'flex';
    }

    // ------------------------------------------------------------------
    // Render span tree
    // ------------------------------------------------------------------

    displayTrace(root) {
        this.logContent.innerHTML = `<div class="trace-tree">${this.renderSpan(root, 0)}</div>`;
        this.bindSpanEvents();
    }

    renderSpan(span, depth) {
        const kind = span.kind || 'unknown';
        const isLLM = kind === 'llm_call';
        const isError = span.status === 'error';
        const hasChildren = span.children && span.children.length > 0;
        const duration = this.formatDuration(span.duration_ms);
        const childCount = this.countDescendants(span);
        const attrs = span.attributes || {};
        const hasLLMData = isLLM && (attrs.request || attrs.response);

        // Build subtitle parts
        const subtitleParts = [];
        if (attrs.goal) subtitleParts.push(attrs.goal);
        if (attrs.tool_name) subtitleParts.push(`tool: ${attrs.tool_name}`);
        if (attrs.model) subtitleParts.push(attrs.model);
        const subtitle = subtitleParts.length ? ` — ${this.escapeHtml(subtitleParts.join(' · '))}` : '';

        // Token badge for LLM calls with captured data
        let tokenBadge = '';
        const tu = attrs.token_usage;
        if (tu?.total_tokens) {
            tokenBadge = `<span class="badge token-badge">${tu.total_tokens} tok</span>`;
        }

        // "no data" badge for LLM calls without captured attributes
        let noDataBadge = '';
        if (isLLM && !hasLLMData) {
            noDataBadge = `<span class="badge muted">no data</span>`;
        }

        // Error badge
        let errorBadge = '';
        if (isError) {
            errorBadge = `<span class="badge error">error</span>`;
        }

        // Children count badge (for non-leaf spans)
        let childBadge = '';
        if (hasChildren) {
            childBadge = `<span class="badge">${childCount} spans</span>`;
        }

        // LLM detail block (only when request/response was captured)
        let llmDetail = '';
        if (hasLLMData) {
            llmDetail = this.renderLLMDetail(span);
        }

        // Error detail
        let errorDetail = '';
        if (span.error) {
            errorDetail = `<div class="error-detail">${this.escapeHtml(span.error)}</div>`;
        }

        // Recursively render children
        let childrenHtml = '';
        if (hasChildren) {
            childrenHtml = span.children.map(c => this.renderSpan(c, depth + 1)).join('');
        }

        const expandable = hasChildren || hasLLMData;
        const expandIcon = expandable ? '<span class="expand-icon">▶</span>' : '<span class="expand-icon-placeholder"></span>';

        return `
        <div class="span-node kind-${kind} ${isError ? 'span-error' : ''}" data-depth="${depth}">
            <div class="span-header ${expandable ? 'expandable' : ''}">
                ${expandIcon}
                <span class="span-kind-badge kind-${kind}">${this.kindLabel(kind)}</span>
                <span class="span-name">${this.escapeHtml(span.name)}</span>
                <span class="span-subtitle">${subtitle}</span>
                <span class="span-duration">${duration}</span>
                ${tokenBadge}${noDataBadge}${errorBadge}${childBadge}
            </div>
            <div class="span-body">
                ${errorDetail}
                ${llmDetail}
                ${childrenHtml ? `<div class="span-children">${childrenHtml}</div>` : ''}
            </div>
        </div>`;
    }

    renderLLMDetail(span) {
        const attrs = span.attributes || {};
        const request = attrs.request;
        const response = attrs.response;

        let html = '<div class="llm-detail">';

        // Request section
        html += `<div class="llm-section request-section">
            <div class="section-header"><span class="expand-icon">▶</span>Request</div>
            <div class="section-content collapsed">${this.formatRequest(request)}</div>
        </div>`;

        // Response section (expanded by default)
        html += `<div class="llm-section response-section">
            <div class="section-header expanded"><span class="expand-icon">▼</span>Response</div>
            <div class="section-content">${this.formatResponse(response)}</div>
        </div>`;

        // Token usage
        const tu = attrs.token_usage;
        if (tu) {
            html += `<div class="token-summary">
                <span>Prompt: <b>${tu.prompt_tokens ?? '—'}</b></span>
                <span>Completion: <b>${tu.completion_tokens ?? '—'}</b></span>
                <span>Total: <b>${tu.total_tokens ?? '—'}</b></span>
            </div>`;
        }

        html += '</div>';
        return html;
    }

    formatRequest(request) {
        if (!request) return '<div class="msg-block no-data">No request data</div>';
        let html = '';

        // Tools provided to the model
        if (request.tools && request.tools.length) {
            html += '<div class="tools-provided">';
            html += '<div class="tools-header">🛠 Tools Provided</div>';
            html += '<div class="tools-list">';
            request.tools.forEach(tool => {
                html += `<div class="tool-def">
                    <span class="tool-def-name">${this.escapeHtml(tool.name)}</span>
                    <span class="tool-def-desc">${this.escapeHtml(tool.description || '')}</span>
                </div>`;
            });
            html += '</div></div>';
        }

        if (request.messages) {
            request.messages.forEach(msg => {
                const role = msg.role || 'unknown';
                const label = this.roleLabel(role);
                const icon = this.roleIcon(role);
                const cls = `msg-block ${role}-message`;
                html += `<div class="${cls}">
                    <div class="msg-header ${role}-header">
                        <span class="role-icon">${icon}</span><strong>${label}</strong>
                        <span class="copy-hint">Double-click to copy</span>
                    </div>
                    <div class="msg-body">${this.escapeHtml(msg.content || '')}</div>
                </div>`;

                // Inline tool calls on AI messages
                if (msg.tool_calls && msg.tool_calls.length) {
                    msg.tool_calls.forEach(tc => {
                        html += `<div class="tool-call-block">
                            <div class="tool-call-header">🔧 Tool Call: <strong>${this.escapeHtml(tc.name)}</strong>
                                ${tc.id ? `<span class="tool-call-id">${this.escapeHtml(tc.id)}</span>` : ''}
                            </div>
                            <pre class="tool-call-args">${this.escapeHtml(JSON.stringify(tc.args, null, 2))}</pre>
                        </div>`;
                    });
                }

                // Tool message metadata
                if (msg.tool_call_id) {
                    html += `<div class="tool-meta">↩ tool_call_id: <code>${this.escapeHtml(msg.tool_call_id)}</code></div>`;
                }
            });
        }
        return html || '<div class="msg-block no-data">No messages</div>';
    }

    formatResponse(response) {
        if (!response) return '<div class="no-data">No response data</div>';
        let html = '';

        // Text content
        if (response.content) {
            html += `<div class="response-body">${this.escapeHtml(response.content)}</div>`;
        }

        // Tool calls selected by the model
        if (response.tool_calls && response.tool_calls.length) {
            html += '<div class="tool-calls-response">';
            response.tool_calls.forEach(tc => {
                html += `<div class="tool-call-block">
                    <div class="tool-call-header">🔧 Tool Call: <strong>${this.escapeHtml(tc.name)}</strong>
                        ${tc.id ? `<span class="tool-call-id">${this.escapeHtml(tc.id)}</span>` : ''}
                    </div>
                    <pre class="tool-call-args">${this.escapeHtml(JSON.stringify(tc.args, null, 2))}</pre>
                </div>`;
            });
            html += '</div>';
        }

        return html || '<div class="no-data">No response data</div>';
    }

    // ------------------------------------------------------------------
    // Event binding
    // ------------------------------------------------------------------

    bindSpanEvents() {
        // Span header expand / collapse
        this.logContent.querySelectorAll('.span-header.expandable').forEach(header => {
            header.addEventListener('click', () => {
                const node = header.closest('.span-node');
                const body = node.querySelector(':scope > .span-body');
                const icon = header.querySelector('.expand-icon');
                const expanded = body.classList.toggle('expanded');
                header.classList.toggle('expanded', expanded);
                icon.textContent = expanded ? '▼' : '▶';
            });
        });

        // Section (Request / Response) toggle
        this.logContent.querySelectorAll('.section-header').forEach(header => {
            header.addEventListener('click', (e) => {
                e.stopPropagation();
                const content = header.nextElementSibling;
                const icon = header.querySelector('.expand-icon');
                const expanded = header.classList.toggle('expanded');
                content.classList.toggle('collapsed', !expanded);
                icon.textContent = expanded ? '▼' : '▶';
            });
        });

        // Double-click to copy
        this.logContent.querySelectorAll('.msg-body, .response-body').forEach(el => {
            el.addEventListener('dblclick', async (e) => {
                e.stopPropagation();
                try {
                    await navigator.clipboard.writeText(el.textContent);
                    this.showCopyFeedback(el);
                } catch (err) { /* ignore */ }
            });
        });
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    kindLabel(kind) {
        const map = {
            session: 'SESSION',
            intent: 'INTENT',
            pursuit: 'PURSUIT',
            tool_call: 'TOOL',
            llm_call: 'LLM',
        };
        return map[kind] || kind.toUpperCase();
    }

    roleLabel(role) {
        const map = { system: 'System Prompt', human: 'User Query', user: 'User Query', ai: 'Assistant', assistant: 'Assistant' };
        return map[role] || role.charAt(0).toUpperCase() + role.slice(1);
    }

    roleIcon(role) {
        const map = { system: '🔧', human: '👤', user: '👤', ai: '🤖', assistant: '🤖' };
        return map[role] || '💬';
    }

    countDescendants(span) {
        let n = 0;
        (span.children || []).forEach(c => { n += 1 + this.countDescendants(c); });
        return n;
    }

    formatDuration(ms) {
        if (ms == null) return '';
        if (ms < 1000) return `${Math.round(ms)}ms`;
        if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
        return `${(ms / 60000).toFixed(1)}m`;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = String(text);
        return div.innerHTML;
    }

    showError(message) {
        this.logContent.innerHTML = `
            <div class="welcome">
                <h2>Error</h2>
                <p style="color: var(--error-color);">${this.escapeHtml(message)}</p>
                <p>Please select a valid trace YAML file.</p>
            </div>`;
        this.stats.style.display = 'none';
    }

    showCopyFeedback(element) {
        const note = document.createElement('div');
        note.className = 'copy-notification';
        note.textContent = '✓ Copied!';
        note.style.cssText = 'position:fixed;top:20px;right:20px;z-index:1000';
        document.body.appendChild(note);

        const origBg = element.style.backgroundColor;
        element.style.backgroundColor = '#d4edda';
        element.style.transition = 'background-color 0.3s ease';
        setTimeout(() => {
            element.style.backgroundColor = origBg;
            note.parentNode?.removeChild(note);
        }, 1500);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.traceViewer = new TraceViewer();
});