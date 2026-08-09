/**
 * BankLoanAgent — SSE Client + Chat UI Logic
 * 使用 fetch + ReadableStream 手动解析 SSE 流（POST 无法用 EventSource）
 */

// ==================== State ====================
let currentThreadId = null;
let isStreaming = false;

// ==================== DOM References ====================
const messagesEl = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const sessionIdEl = document.getElementById('session-id');
const userIdInput = document.getElementById('user-id-input');
const toolPanel = document.getElementById('tool-panel');
const toolPanelBody = document.getElementById('tool-panel-body');
const handoffModal = document.getElementById('handoff-modal');

// ==================== Core: Send Message + SSE Stream ====================

async function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isStreaming) return;

    // 清除欢迎语
    const welcome = messagesEl.querySelector('.welcome-message');
    if (welcome) welcome.remove();

    // 渲染用户消息
    appendMessage('user', text);
    messageInput.value = '';
    messageInput.disabled = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '回复中...';

    // 创建 assistant 消息占位
    const assistantEl = appendMessage('assistant', '');
    const contentEl = assistantEl.querySelector('.message-content');
    showCursor(contentEl);

    isStreaming = true;

    try {
        const userId = userIdInput.value.trim() || 'anonymous';

        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                thread_id: currentThreadId,
                user_id: userId
            })
        });

        // 从响应头获取 thread_id
        currentThreadId = resp.headers.get('X-Thread-Id') || currentThreadId;
        updateSessionDisplay();

        // ── 手动解析 SSE 流 ──
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE 帧以 \n\n 分隔
            const frames = buffer.split('\n\n');
            buffer = frames.pop(); // 保留未完成的帧

            for (const frame of frames) {
                if (!frame.trim()) continue;
                const parsed = parseSSEFrame(frame);
                if (parsed) {
                    handleEvent(parsed.event, parsed.data, contentEl);
                }
            }
        }
    } catch (err) {
        appendMessage('system-error', `连接错误: ${err.message}`);
    } finally {
        isStreaming = false;
        hideCursor(contentEl);
        messageInput.disabled = false;
        sendBtn.disabled = false;
        sendBtn.textContent = '发送';
        messageInput.focus();
    }
}

// ==================== SSE Frame Parser ====================

function parseSSEFrame(frame) {
    let eventType = '';
    let dataStr = '';

    for (const line of frame.split('\n')) {
        if (line.startsWith('event: ')) {
            eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
            dataStr = line.slice(6);
        }
    }

    if (!eventType || !dataStr) return null;

    try {
        return { event: eventType, data: JSON.parse(dataStr) };
    } catch (e) {
        console.error('Failed to parse SSE data:', dataStr, e);
        return null;
    }
}

// ==================== Event Handler ====================

function handleEvent(eventType, data, contentEl) {
    switch (eventType) {
        case 'token':
            contentEl.textContent += data.content;
            scrollToBottom();
            break;

        case 'tool_start':
            showToolPanel();
            appendToolItem(data.name, 'running');
            break;

        case 'tool_end':
            updateToolItem(data.name, 'done', data.output);
            break;

        case 'handoff':
            showHandoffModal(data.message || '需要转接人工客服', data.thread_id);
            break;

        case 'done':
            // 流式完成，cursor 在 finally 中隐藏
            break;

        case 'error':
            appendMessage('system-error', data.message || '未知错误');
            break;
    }
}

// ==================== Handoff Resolve ====================

async function resolveHandoff(action) {
    const content = document.getElementById('handoff-reply-content').value.trim();
    const threadId = document.getElementById('handoff-thread-id').textContent;

    if (action === 'reply' && !content) {
        alert('请输入回复内容');
        return;
    }

    try {
        const resp = await fetch('/api/handoff/resolve', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                thread_id: threadId,
                action: action,
                content: content
            })
        });

        if (resp.ok) {
            closeHandoffModal();
            appendMessage('user', `[人工回复] ${action === 'reply' ? content : action}`);
            // 加载历史获取图继续执行后的结果
            await loadHistory(threadId);
        } else {
            const err = await resp.json();
            alert('处理失败: ' + (err.detail || '未知错误'));
        }
    } catch (err) {
        alert('请求错误: ' + err.message);
    }
}

// ==================== Session ====================

async function createSession() {
    try {
        const resp = await fetch('/api/session', { method: 'POST' });
        const data = await resp.json();
        currentThreadId = data.thread_id;
        updateSessionDisplay();
        messagesEl.innerHTML = '';
        messagesEl.innerHTML = '<div class="welcome-message"><p>👋 新会话已创建，请输入您的问题。</p></div>';
    } catch (err) {
        alert('创建会话失败: ' + err.message);
    }
}

// ==================== History ====================

async function loadHistory(threadId) {
    try {
        const resp = await fetch(`/api/history/${threadId}`);
        const data = await resp.json();

        // 只追加最新的 assistant 消息
        const msgs = data.messages || [];
        if (msgs.length > 0) {
            const lastMsg = msgs[msgs.length - 1];
            if (lastMsg.role === 'assistant') {
                appendMessage('assistant', lastMsg.content);
                scrollToBottom();
            }
        }
    } catch (err) {
        console.error('Failed to load history:', err);
    }
}

// ==================== UI Helpers ====================

function appendMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const contentSpan = document.createElement('span');
    contentSpan.className = 'message-content';
    contentSpan.textContent = content;

    bubble.appendChild(contentSpan);
    div.appendChild(bubble);
    messagesEl.appendChild(div);
    scrollToBottom();

    return div;
}

function showCursor(contentEl) {
    contentEl.classList.add('streaming-cursor');
}

function hideCursor(contentEl) {
    contentEl.classList.remove('streaming-cursor');
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function updateSessionDisplay() {
    sessionIdEl.textContent = currentThreadId
        ? currentThreadId.substring(0, 8) + '...'
        : '—';
}

// ==================== Tool Panel ====================

function showToolPanel() {
    toolPanel.style.display = '';
}

function appendToolItem(toolName, status) {
    const item = document.createElement('div');
    item.className = 'tool-item';
    item.id = `tool-${toolName}`;
    item.innerHTML = `
        <span class="tool-status">⏳</span>
        <span class="tool-name">${toolName}</span>
    `;
    toolPanelBody.appendChild(item);
}

function updateToolItem(toolName, status, output) {
    const item = document.getElementById(`tool-${toolName}`);
    if (!item) {
        appendToolItem(toolName, status);
        return;
    }
    const statusEl = item.querySelector('.tool-status');
    statusEl.textContent = '✅';

    if (output) {
        const outSpan = document.createElement('span');
        outSpan.className = 'tool-output';
        outSpan.textContent = output;
        item.appendChild(outSpan);
    }
}

function toggleToolPanel() {
    toolPanelBody.classList.toggle('collapsed');
    const icon = document.getElementById('tool-toggle');
    icon.textContent = toolPanelBody.classList.contains('collapsed') ? '▶' : '▼';
}

// ==================== Handoff Modal ====================

function showHandoffModal(message, threadId) {
    document.getElementById('handoff-message').textContent = message;
    document.getElementById('handoff-thread-id').textContent = threadId;
    document.getElementById('handoff-reply-content').value = '';
    handoffModal.style.display = 'flex';
}

function closeHandoffModal() {
    handoffModal.style.display = 'none';
}

// ==================== Init ====================

messageInput.focus();
