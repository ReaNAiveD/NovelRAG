/**
 * LLM Log Viewer - JavaScript functionality
 * Handles YAML log file parsing, directory browsing, and UI interactions
 */

class LLMLogViewer {
    constructor() {
        this.currentLog = null;
        this.currentDirectory = null;
        this.selectedFile = null;
        this.files = new Map(); // filename -> File object
        
        this.initializeElements();
        this.bindEvents();
    }
    
    initializeElements() {
        // File inputs
        this.fileInput = document.getElementById('fileInput');
        this.singleFileInput = document.getElementById('singleFileInput');
        this.fileName = document.getElementById('fileName');
        
        // Directory panel
        this.directoryPath = document.getElementById('directoryPath');
        this.directoryContent = document.getElementById('directoryContent');
        
        // Stats
        this.stats = document.getElementById('stats');
        this.pursuitCount = document.getElementById('pursuitCount');
        this.totalCalls = document.getElementById('totalCalls');
        this.uniqueTemplates = document.getElementById('uniqueTemplates');
        
        // Log content
        this.logContent = document.getElementById('logContent');
        
        // Show welcome message initially
        this.showWelcomeMessage();
    }
    
    bindEvents() {
        // Directory selection (multiple files)
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.loadDirectory(e.target.files);
            }
        });
        
        // Single file selection
        this.singleFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.loadSingleFile(e.target.files[0]);
            }
        });
    }
    
    loadDirectory(fileList) {
        this.files.clear();
        const yamlFiles = Array.from(fileList).filter(file => 
            file.name.toLowerCase().endsWith('.yaml') || 
            file.name.toLowerCase().endsWith('.yml')
        );
        
        if (yamlFiles.length === 0) {
            this.fileName.textContent = 'No YAML files found in directory';
            this.showDirectoryEmpty('No YAML files found');
            return;
        }
        
        // Store files and get directory path
        yamlFiles.forEach(file => {
            this.files.set(file.name, file);
        });
        
        // Extract directory path from first file
        const firstFile = yamlFiles[0];
        const relativePath = firstFile.webkitRelativePath || firstFile.name;
        const directoryPath = relativePath.includes('/') ? 
            relativePath.substring(0, relativePath.lastIndexOf('/')) : 
            'Selected Directory';
            
        this.currentDirectory = directoryPath;
        this.directoryPath.textContent = directoryPath;
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
        // Sort files by name
        files.sort((a, b) => a.name.localeCompare(b.name));
        
        this.directoryContent.innerHTML = files.map(file => `
            <div class="file-item" data-filename="${file.name}">
                <span class="file-icon">📄</span>
                <span class="file-name-text">${file.name}</span>
                <span class="file-size">${this.formatFileSize(file.size)}</span>
            </div>
        `).join('');
        
        // Bind click events to file items
        this.directoryContent.querySelectorAll('.file-item').forEach(item => {
            item.addEventListener('click', () => {
                const filename = item.dataset.filename;
                this.selectFile(filename);
            });
        });
    }
    
    showDirectoryEmpty(message = 'No directory selected') {
        this.directoryContent.innerHTML = `
            <div class="directory-empty">
                <p>${message}</p>
                <p class="help-text">Use "Select Directory" button to browse YAML log files</p>
            </div>
        `;
    }
    
    async selectFile(filename) {
        // Update UI selection
        this.directoryContent.querySelectorAll('.file-item').forEach(item => {
            item.classList.remove('selected');
            if (item.dataset.filename === filename) {
                item.classList.add('selected');
            }
        });
        
        this.selectedFile = filename;
        
        // Load and display the file
        const file = this.files.get(filename);
        if (file) {
            try {
                const content = await this.readFileContent(file);
                this.parseAndDisplayLog(content, filename);
            } catch (error) {
                console.error('Error reading file:', error);
                this.showError(`Error reading file: ${error.message}`);
            }
        }
    }
    
    readFileContent(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }
    
    parseAndDisplayLog(content, filename) {
        try {
            // Parse YAML content
            const logData = jsyaml.load(content);
            
            if (!logData || !logData.pursuits || !Array.isArray(logData.pursuits)) {
                throw new Error('Invalid log format: missing or invalid pursuits array');
            }
            
            this.currentLog = logData;
            this.updateStats();
            this.displayPursuits();
            
        } catch (error) {
            console.error('Error parsing YAML:', error);
            this.showError(`Error parsing ${filename}: ${error.message}`);
        }
    }
    
    updateStats() {
        if (!this.currentLog) return;
        
        const pursuits = this.currentLog.pursuits;
        const totalCalls = pursuits.reduce((sum, p) => sum + (p.llm_calls?.length || 0), 0);
        const allTemplates = new Set();
        
        pursuits.forEach(pursuit => {
            if (pursuit.llm_calls) {
                pursuit.llm_calls.forEach(call => {
                    if (call.template_name) {
                        allTemplates.add(call.template_name);
                    }
                });
            }
        });
        
        this.pursuitCount.textContent = pursuits.length;
        this.totalCalls.textContent = totalCalls;
        this.uniqueTemplates.textContent = allTemplates.size;
        
        this.stats.style.display = 'flex';
    }
    
    displayPursuits() {
        if (!this.currentLog) return;
        
        const pursuits = this.currentLog.pursuits;
        
        this.logContent.innerHTML = `
            <div class="log-entries">
                ${pursuits.map((pursuit, index) => this.renderPursuit(pursuit, index)).join('')}
            </div>
        `;
        
        this.bindPursuitEvents();
    }
    
    renderPursuit(pursuit, index) {
        const duration = pursuit.completed_at ? 
            this.calculateDuration(pursuit.started_at, pursuit.completed_at) : 'N/A';
        const callCount = pursuit.llm_calls?.length || 0;
        
        return `
            <div class="pursuit" data-index="${index}">
                <div class="pursuit-header">
                    <div class="pursuit-title">
                        <span class="expand-icon">▶</span>
                        Goal: ${this.escapeHtml(pursuit.goal)}
                        <span class="duration-info">(${duration})</span>
                        <span class="badge">${callCount} calls</span>
                    </div>
                </div>
                <div class="pursuit-content">
                    ${pursuit.llm_calls ? pursuit.llm_calls.map((call, callIndex) => 
                        this.renderLLMCall(call, callIndex)).join('') : ''}
                </div>
            </div>
        `;
    }
    
    renderLLMCall(call, index) {
        const duration = call.duration_ms ? `${call.duration_ms}ms` : 'N/A';
        
        return `
            <div class="llm-call" data-index="${index}">
                <div class="llm-call-header">
                    <div class="llm-call-title">
                        <span class="expand-icon">▶</span>
                        ${this.escapeHtml(call.template_name)}
                        <span class="duration-info">(${duration})</span>
                    </div>
                </div>
                <div class="llm-call-content">
                    <div class="request-section">
                        <div class="section-header">
                            <span class="expand-icon">▶</span>
                            Request
                        </div>
                        <div class="section-content collapsed">${this.formatRequestAsBlocks(call.request)}</div>
                    </div>
                    <div class="response-section">
                        <div class="section-header expanded">
                            <span class="expand-icon">▼</span>
                            Response
                        </div>
                        <div class="section-content">${this.formatResponse(call.response)}</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    formatRequestAsBlocks(request) {
        if (!request) return '<div class="request-block no-data">No request data</div>';
        
        let html = '';
        
        if (request.messages) {
            request.messages.forEach((msg, i) => {
                const roleClass = `request-block ${msg.role}-message`;
                const roleLabel = msg.role.charAt(0).toUpperCase() + msg.role.slice(1);
                
                if (msg.role === 'system') {
                    // System message gets special prominent treatment
                    html += `<div class="${roleClass} system-prompt"><div class="block-header system-header"><span class="role-icon">🔧</span><strong>${roleLabel} Prompt</strong><span class="copy-hint">Double-click to copy</span></div><div class="block-content">${this.escapeHtml(msg.content)}</div></div>`;
                } else if (msg.role === 'user') {
                    // User message gets secondary treatment
                    html += `<div class="${roleClass} user-query"><div class="block-header user-header"><span class="role-icon">👤</span><strong>${roleLabel} Query</strong><span class="copy-hint">Double-click to copy</span></div><div class="block-content">${this.escapeHtml(msg.content)}</div></div>`;
                } else {
                    // Other roles (assistant, etc.)
                    html += `<div class="${roleClass}"><div class="block-header other-header"><span class="role-icon">🤖</span><strong>${roleLabel}</strong></div><div class="block-content">${this.escapeHtml(msg.content)}</div></div>`;
                }
            });
        }
        
        // Add other request parameters as a separate reference block
        const otherParams = Object.keys(request).filter(key => key !== 'messages');
        if (otherParams.length > 0) {
            html += `<div class="request-block parameters-block"><div class="block-header params-header"><span class="role-icon">⚙️</span><strong>Request Parameters</strong><span class="copy-hint">Reference info</span></div><div class="block-content params-content">`;
            
            otherParams.forEach(key => {
                const value = request[key];
                if (value !== null && value !== undefined) {
                    html += `<div class="param-item"><span class="param-key">${key}:</span> <span class="param-value">${this.escapeHtml(JSON.stringify(value))}</span></div>`;
                }
            });
            
            html += `</div></div>`;
        }
        
        return html;
    }
    
    formatRequest(request) {
        if (!request) return 'No request data';
        
        let formatted = '';
        
        if (request.messages) {
            request.messages.forEach((msg, i) => {
                const roleLabel = msg.role.charAt(0).toUpperCase() + msg.role.slice(1);
                
                if (msg.role === 'system') {
                    // System message gets special treatment - most important
                    formatted += `=== ${roleLabel} Prompt ===\n`;
                    formatted += `${msg.content}\n\n`;
                } else if (msg.role === 'user') {
                    // User message is secondary but still important
                    formatted += `--- ${roleLabel} Query ---\n`;
                    formatted += `${msg.content}\n\n`;
                } else {
                    // Other roles (assistant, etc.) - less common
                    formatted += `[${roleLabel}]: ${msg.content}\n\n`;
                }
            });
        }
        
        // Add other request parameters as reference info
        const otherParams = Object.keys(request).filter(key => key !== 'messages');
        if (otherParams.length > 0) {
            formatted += '--- Request Parameters (Reference) ---\n';
            otherParams.forEach(key => {
                const value = request[key];
                if (value !== null && value !== undefined) {
                    if (typeof value === 'string' && value.length > 100) {
                        // Long strings get abbreviated in reference section
                        formatted += `${key}: ${value.substring(0, 100)}...\n`;
                    } else {
                        formatted += `${key}: ${JSON.stringify(value)}\n`;
                    }
                }
            });
        }
        
        return this.escapeHtml(formatted.trim());
    }
    
    formatResponse(response) {
        if (!response) return 'No response data';
        
        let formatted = response.content || '';
        
        // Add other response metadata if available
        const otherParams = Object.keys(response).filter(key => key !== 'content');
        if (otherParams.length > 0) {
            if (formatted) formatted += '\n\n';
            formatted += '--- Response Metadata ---\n';
            otherParams.forEach(key => {
                const value = response[key];
                if (value !== null && value !== undefined) {
                    formatted += `${key}: ${value}\n`;
                }
            });
        }
        
        return this.escapeHtml(formatted.trim());
    }
    
    bindPursuitEvents() {
        // Pursuit expand/collapse
        this.logContent.querySelectorAll('.pursuit-header').forEach(header => {
            header.addEventListener('click', () => {
                const pursuit = header.parentElement;
                const content = pursuit.querySelector('.pursuit-content');
                const icon = header.querySelector('.expand-icon');
                
                header.classList.toggle('expanded');
                content.classList.toggle('expanded');
                icon.textContent = content.classList.contains('expanded') ? '▼' : '▶';
            });
        });
        
        // LLM call expand/collapse
        this.logContent.querySelectorAll('.llm-call-header').forEach(header => {
            header.addEventListener('click', () => {
                const call = header.parentElement;
                const content = call.querySelector('.llm-call-content');
                const icon = header.querySelector('.expand-icon');
                
                header.classList.toggle('expanded');
                content.classList.toggle('expanded');
                icon.textContent = content.classList.contains('expanded') ? '▼' : '▶';
            });
        });
        
        // Section expand/collapse (Request/Response)
        this.logContent.querySelectorAll('.section-header').forEach(header => {
            header.addEventListener('click', () => {
                const content = header.nextElementSibling;
                const icon = header.querySelector('.expand-icon');
                
                header.classList.toggle('expanded');
                content.classList.toggle('collapsed');
                icon.textContent = header.classList.contains('expanded') ? '▼' : '▶';
            });
        });
        
        // Add double-click to copy functionality
        this.logContent.querySelectorAll('.section-content').forEach(content => {
            // Add double-click to copy to clipboard
            content.addEventListener('dblclick', async (e) => {
                const requestBlock = e.target.closest('.request-block');
                let textToCopy;
                
                if (requestBlock) {
                    const blockContent = requestBlock.querySelector('.block-content');
                    textToCopy = blockContent ? blockContent.textContent : content.textContent;
                } else {
                    textToCopy = content.textContent;
                }
                
                try {
                    await navigator.clipboard.writeText(textToCopy);
                    this.showCopyFeedback(e.target.closest('.request-block') || content);
                } catch (err) {
                    console.log('Copy to clipboard failed:', err);
                }
            });
        });
        
        // Add specific handlers for request blocks
        this.logContent.querySelectorAll('.block-content').forEach(blockContent => {
            blockContent.addEventListener('dblclick', async (e) => {
                e.stopPropagation();
                try {
                    await navigator.clipboard.writeText(blockContent.textContent);
                    this.showCopyFeedback(blockContent.closest('.request-block'));
                } catch (err) {
                    console.log('Copy to clipboard failed:', err);
                }
            });
        });
    }
    
    showWelcomeMessage() {
        this.logContent.innerHTML = `
            <div class="welcome">
                <h2>Welcome to LLM Log Viewer</h2>
                <p>Select a directory or YAML log file to view LLM request/response history from agent pursuits.</p>
                <ul>
                    <li>Use "Select Directory" to browse multiple YAML files</li>
                    <li>Use "Select File" to load a single YAML file</li>
                    <li>Each pursuit shows the goal and can be expanded to view LLM calls</li>
                    <li>Each LLM call shows the template used and can be expanded to view details</li>
                    <li>Requests are collapsed by default, responses are expanded</li>
                </ul>
            </div>
        `;
        this.stats.style.display = 'none';
    }
    
    showError(message) {
        this.logContent.innerHTML = `
            <div class="welcome">
                <h2>Error</h2>
                <p style="color: var(--error-color);">${this.escapeHtml(message)}</p>
                <p>Please select a valid YAML log file.</p>
            </div>
        `;
        this.stats.style.display = 'none';
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }
    
    calculateDuration(startTime, endTime) {
        if (!endTime) return 'N/A';
        
        const start = new Date(startTime);
        const end = new Date(endTime);
        const diff = end - start;
        
        if (diff < 1000) return `${diff}ms`;
        if (diff < 60000) return `${(diff / 1000).toFixed(1)}s`;
        return `${(diff / 60000).toFixed(1)}m`;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showCopyFeedback(element) {
        const originalBg = element.style.backgroundColor;
        element.style.backgroundColor = '#d4edda';
        element.style.transition = 'background-color 0.3s ease';
        setTimeout(() => {
            element.style.backgroundColor = originalBg;
        }, 300);
    }
}

// Initialize the log viewer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.logViewer = new LLMLogViewer();
});