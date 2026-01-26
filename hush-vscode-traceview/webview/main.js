// @ts-check

const vscode = acquireVsCodeApi();

let currentTraces = [];
let currentNodes = [];
let selectedNode = null;
let currentView = 'list';
let expandedNodes = new Set();

// Request initial data
vscode.postMessage({ type: 'getTraceList' });
vscode.postMessage({ type: 'getDbInfo' });

// Handle messages from extension
window.addEventListener('message', event => {
    const message = event.data;
    switch (message.type) {
        case 'traceList':
            currentTraces = message.traces || [];
            if (currentView === 'list') {
                renderTraceList(message.traces, message.error);
            }
            break;
        case 'traceDetail':
            currentNodes = message.nodes || [];
            renderTraceDetail(message.requestId, message.nodes, message.error);
            break;
        case 'dbInfo':
            renderDbInfo(message);
            break;
    }
});

function refresh() {
    vscode.postMessage({ type: 'getTraceList' });
    vscode.postMessage({ type: 'getDbInfo' });
}

function clearTraces() {
    vscode.postMessage({ type: 'clearTraces' });
}

function renderDbInfo(info) {
    const el = document.getElementById('db-info');
    if (!el) return;
    const sizeKb = (info.size / 1024).toFixed(1);
    el.textContent = `Database: ${info.path} (${info.exists ? sizeKb + ' KB' : 'not found'})`;
}

function renderTraceList(traces, error) {
    currentView = 'list';
    const listEl = document.getElementById('trace-list');
    const detailEl = document.getElementById('trace-detail');

    if (!listEl || !detailEl) return;

    listEl.style.display = 'block';
    detailEl.style.display = 'none';

    if (error) {
        listEl.innerHTML = `<div class="error-state">Error: ${escapeHtml(error)}</div>`;
        return;
    }

    if (!traces || traces.length === 0) {
        listEl.innerHTML = '<div class="empty-state">No traces found.<br><br>Run a workflow with a tracer to see traces here.</div>';
        return;
    }

    let html = `
        <table class="trace-table">
            <thead>
                <tr>
                    <th>Workflow</th>
                    <th>Time</th>
                    <th>Duration</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                </tr>
            </thead>
            <tbody>
    `;

    for (const trace of traces) {
        const time = trace.start_time ? formatTime(trace.start_time) : '-';
        const duration = trace.total_duration_ms ? formatDuration(trace.total_duration_ms) : '-';
        const tokens = trace.total_tokens || '-';
        const cost = trace.total_cost ? '$' + trace.total_cost.toFixed(4) : '-';

        html += `
            <tr onclick="showTrace('${escapeHtml(trace.request_id)}')">
                <td class="workflow">${escapeHtml(trace.workflow_name)}</td>
                <td class="time">${time}</td>
                <td class="duration">${duration}</td>
                <td class="tokens">${tokens}</td>
                <td class="cost">${cost}</td>
            </tr>
        `;
    }

    html += '</tbody></table>';
    listEl.innerHTML = html;
}

function showTrace(requestId) {
    vscode.postMessage({ type: 'getTraceDetail', requestId });
}

function renderTraceDetail(requestId, nodes, error) {
    currentView = 'detail';
    expandedNodes = new Set();

    const listEl = document.getElementById('trace-list');
    const detailEl = document.getElementById('trace-detail');

    if (!listEl || !detailEl) return;

    listEl.style.display = 'none';
    detailEl.style.display = 'grid';

    if (error) {
        detailEl.innerHTML = `<div class="error-state">Error: ${escapeHtml(error)}</div>`;
        return;
    }

    const trace = currentTraces.find(t => t.request_id === requestId);
    const workflowName = trace ? trace.workflow_name : 'Unknown';
    const time = trace && trace.start_time ? formatTime(trace.start_time) : '';

    // Expand all nodes by default
    expandAllNodes(nodes);

    let html = `
        <div id="detail-header">
            <button id="back-btn" onclick="backToList()">← Back</button>
            <span class="detail-workflow">${escapeHtml(workflowName)}</span>
            <span class="detail-time">${time}</span>
        </div>
        <div id="tree-panel">
            <div class="tree-root">
                ${renderTree(nodes)}
            </div>
        </div>
        <div id="node-panel">
            <div class="empty-state">Select a node to view details</div>
        </div>
    `;

    detailEl.innerHTML = html;

    // Select first node by default
    if (nodes.length > 0) {
        selectNode(nodes[0]);
    }
}

function expandAllNodes(nodes) {
    for (const node of nodes) {
        if (node.children && node.children.length > 0) {
            expandedNodes.add(node.id);
            expandAllNodes(node.children);
        }
    }
}

function renderTree(nodes) {
    let html = '';
    for (const node of nodes) {
        html += renderTreeNode(node);
    }
    return html;
}

function renderTreeNode(node) {
    const hasChildren = node.children && node.children.length > 0;
    const isExpanded = expandedNodes.has(node.id);
    const icon = getNodeIcon(node);
    const duration = node.duration_ms ? formatDuration(node.duration_ms) : '';
    const tokens = node.total_tokens ? `${node.total_tokens}` : '';

    let html = `
        <div class="tree-node">
            <div class="tree-item" data-id="${node.id}" onclick="selectNodeById(${node.id})">
                <span class="tree-toggle ${hasChildren ? 'has-children' : ''}" onclick="event.stopPropagation(); toggleNode(${node.id})">
                    ${hasChildren ? (isExpanded ? '▼' : '▶') : ''}
                </span>
                <span class="tree-icon">${icon}</span>
                <span class="tree-name">${escapeHtml(getShortName(node.node_name))}</span>
                <span class="tree-info">
                    ${duration ? `<span class="tree-duration">${duration}</span>` : ''}
                    ${tokens ? `<span class="tree-tokens">${tokens}</span>` : ''}
                </span>
            </div>
    `;

    if (hasChildren) {
        html += `<div class="tree-children" style="display: ${isExpanded ? 'block' : 'none'}">`;
        for (const child of node.children) {
            html += renderTreeNode(child);
        }
        html += '</div>';
    }

    html += '</div>';
    return html;
}

function toggleNode(id) {
    if (expandedNodes.has(id)) {
        expandedNodes.delete(id);
    } else {
        expandedNodes.add(id);
    }

    // Re-render the tree
    const treePanel = document.getElementById('tree-panel');
    if (treePanel) {
        treePanel.innerHTML = `<div class="tree-root">${renderTree(currentNodes)}</div>`;

        // Re-apply selection
        if (selectedNode) {
            const el = document.querySelector(`.tree-item[data-id="${selectedNode.id}"]`);
            if (el) {
                el.classList.add('selected');
            }
        }
    }
}

function getNodeIcon(node) {
    if (node.contain_generation) return '⭐';
    if (node.node_name && node.node_name.includes('iteration[')) return '📂';
    if (node.node_name && (node.node_name.includes('_loop') || node.node_name.includes('map_'))) return '🔄';
    if (node.children && node.children.length > 0) return '📦';
    return '○';
}

function getShortName(name) {
    if (!name) return 'unknown';
    // Get last part after dot, but keep iteration suffix
    const match = name.match(/([^.]+)(\.\w+\[\d+\])?$/);
    if (match) {
        return match[1] + (match[2] || '');
    }
    return name;
}

function selectNodeById(id) {
    const node = findNodeById(currentNodes, id);
    if (node) {
        selectNode(node);
    }
}

function findNodeById(nodes, id) {
    for (const node of nodes) {
        if (node.id === id) return node;
        if (node.children) {
            const found = findNodeById(node.children, id);
            if (found) return found;
        }
    }
    return null;
}

function selectNode(node) {
    selectedNode = node;

    // Update selection UI
    document.querySelectorAll('.tree-item').forEach(el => {
        el.classList.remove('selected');
    });
    const el = document.querySelector(`.tree-item[data-id="${node.id}"]`);
    if (el) {
        el.classList.add('selected');
    }

    renderNodeDetail(node);
}

function renderNodeDetail(node) {
    const panel = document.getElementById('node-panel');
    if (!panel) return;

    const duration = node.duration_ms ? formatDuration(node.duration_ms) : '-';
    const model = node.model || '-';
    const promptTokens = node.prompt_tokens || 0;
    const completionTokens = node.completion_tokens || 0;
    const tokens = node.total_tokens ? `${promptTokens} → ${completionTokens}` : '-';
    const cost = node.cost_usd ? '$' + node.cost_usd.toFixed(4) : '-';

    let html = `
        <div class="node-title">${escapeHtml(node.node_name)}</div>
        <div class="node-metrics">
            <div class="metric">
                <div class="metric-label">Duration</div>
                <div class="metric-value">${duration}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Model</div>
                <div class="metric-value">${escapeHtml(model)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Tokens</div>
                <div class="metric-value">${tokens}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Cost</div>
                <div class="metric-value">${cost}</div>
            </div>
        </div>
    `;

    if (node.input) {
        html += `
            <div class="node-section">
                <div class="node-section-title">Input</div>
                <div class="json-view">${formatJson(node.input)}</div>
            </div>
        `;
    }

    if (node.output) {
        html += `
            <div class="node-section">
                <div class="node-section-title">Output</div>
                <div class="json-view">${formatJson(node.output)}</div>
            </div>
        `;
    }

    if (node.metadata) {
        html += `
            <div class="node-section">
                <div class="node-section-title">Metadata</div>
                <div class="json-view">${formatJson(node.metadata)}</div>
            </div>
        `;
    }

    panel.innerHTML = html;
}

function backToList() {
    currentView = 'list';
    selectedNode = null;
    renderTraceList(currentTraces, null);
}

// Utility functions
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function formatTime(isoString) {
    try {
        const date = new Date(isoString);
        return date.toLocaleString();
    } catch {
        return isoString;
    }
}

function formatDuration(ms) {
    if (ms < 1000) {
        return ms.toFixed(0) + 'ms';
    }
    return (ms / 1000).toFixed(2) + 's';
}

function formatJson(obj) {
    if (!obj) return '';
    try {
        const json = typeof obj === 'string' ? obj : JSON.stringify(obj, null, 2);
        return syntaxHighlight(json);
    } catch {
        return escapeHtml(String(obj));
    }
}

function syntaxHighlight(json) {
    // Escape HTML first
    json = escapeHtml(json);

    // Apply syntax highlighting
    return json.replace(
        /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g,
        function (match) {
            let cls = 'json-number';
            if (/^"/.test(match)) {
                if (/:$/.test(match)) {
                    cls = 'json-key';
                    match = match.slice(0, -1) + '</span>:';
                    return '<span class="' + cls + '">' + match;
                } else {
                    cls = 'json-string';
                }
            } else if (/true|false/.test(match)) {
                cls = 'json-boolean';
            } else if (/null/.test(match)) {
                cls = 'json-null';
            }
            return '<span class="' + cls + '">' + match + '</span>';
        }
    );
}
