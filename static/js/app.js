const chatContainer = document.getElementById('chat-container');
const statusText = document.getElementById('status-text');
const modelSelector = document.getElementById('model-selector');
const wakeBtn = document.getElementById('wake-btn');
const holdBtn = document.getElementById('hold-btn');
const stopBtn = document.getElementById('stop-btn');
const thinkingCheck = document.getElementById('thinking-check');
const searchCheck = document.getElementById('search-check');
const thinkingLabel = document.getElementById('thinking-label');
const searchLabel = document.getElementById('search-label');
const searchIcon = document.getElementById('search-icon');

let ws = null;
let capabilities = {};
let searchIconTimer = null;
let holdActive = false;

const wsUrl = `ws://${window.location.host}/ws`;

const dotColors = {
    listening: '#4caf50',
    transcribing: '#ff9800',
    speaking: '#2196f3',
};

function applyStatus(data) {
    statusText.innerText = data.text;
    wakeBtn.disabled = !data.wake_enabled;
    wakeBtn.classList.toggle('listening', data.phase === 'listening');
    holdBtn.classList.toggle('visible', !!data.record_hold_enabled);
    stopBtn.classList.toggle('visible', !!data.stop_enabled);
    if (!data.record_hold_enabled && holdActive) {
        endHoldRecord();
    }
    if (data.phase === 'sleeping') {
        hideSearchIcon();
    }
    document.querySelector('.status-dot').style.backgroundColor =
        dotColors[data.phase] || '#00bcd4';
    void document.querySelector('.header').offsetHeight;
}

function sendRecordHold(active) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: 'record_hold', active: active }));
}

function endHoldRecord() {
    if (!holdActive) return;
    holdActive = false;
    holdBtn.classList.remove('holding');
    sendRecordHold(false);
}

function startHoldRecord(event) {
    if (!holdBtn.classList.contains('visible')) return;
    event.preventDefault();
    if (holdActive) return;
    holdActive = true;
    holdBtn.classList.add('holding');
    if (holdBtn.setPointerCapture && event.pointerId !== undefined) {
        holdBtn.setPointerCapture(event.pointerId);
    }
    sendRecordHold(true);
}

holdBtn.addEventListener('pointerdown', startHoldRecord);

holdBtn.addEventListener('pointerup', function(event) {
    event.preventDefault();
    endHoldRecord();
});

holdBtn.addEventListener('pointercancel', function(event) {
    event.preventDefault();
    endHoldRecord();
});

window.addEventListener('pointerup', function() {
    endHoldRecord();
});

wakeBtn.addEventListener('click', function() {
    if (ws && ws.readyState === WebSocket.OPEN && !wakeBtn.disabled) {
        ws.send(JSON.stringify({ type: 'wake' }));
    }
});

stopBtn.addEventListener('click', function() {
    if (ws && ws.readyState === WebSocket.OPEN && stopBtn.classList.contains('visible')) {
        ws.send(JSON.stringify({ type: 'stop' }));
    }
});

function sendOptions() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({
        type: 'set_options',
        enable_thinking: thinkingCheck.checked,
        enable_search: searchCheck.checked,
    }));
}

function applyCapabilities(caps) {
    capabilities = caps || {};
    const thinkingSupported = capabilities.supports_thinking !== false;
    const searchSupported =
        capabilities.supports_tool_search || capabilities.supports_native_search;

    thinkingCheck.disabled = !thinkingSupported;
    searchCheck.disabled = !searchSupported;
    thinkingLabel.classList.toggle('disabled', !thinkingSupported);
    searchLabel.classList.toggle('disabled', !searchSupported);

    if (capabilities.thinking_search_exclusive &&
        thinkingCheck.checked && searchCheck.checked) {
        thinkingCheck.checked = false;
        sendOptions();
    }
}

function applyConfig(data) {
    if (data.model) modelSelector.value = data.model;
    if (typeof data.enable_thinking === 'boolean') {
        thinkingCheck.checked = data.enable_thinking;
    }
    if (typeof data.enable_search === 'boolean') {
        searchCheck.checked = data.enable_search;
    }
    if (data.capabilities) applyCapabilities(data.capabilities);
}

function hideSearchIcon() {
    searchIcon.className = 'search-icon';
    searchIcon.textContent = '';
    searchIcon.title = '';
    searchIcon.setAttribute('aria-hidden', 'true');
}

function showSearchIcon(status) {
    if (searchIconTimer) {
        clearTimeout(searchIconTimer);
        searchIconTimer = null;
    }

    searchIcon.className = 'search-icon visible';
    searchIcon.setAttribute('aria-hidden', 'false');

    if (status === 'pending') {
        searchIcon.classList.add('pending');
        searchIcon.textContent = '';
        searchIcon.title = '联网搜索中…';
        return;
    }

    searchIcon.classList.remove('pending');

    if (status === 'success') {
        searchIcon.classList.add('success');
        searchIcon.textContent = '✓';
        searchIcon.title = '联网成功';
    } else if (status === 'failed' || status === 'timeout') {
        searchIcon.classList.add('failed');
        searchIcon.textContent = '✗';
        searchIcon.title = status === 'timeout' ? '联网超时' : '联网失败';
    } else {
        hideSearchIcon();
        return;
    }

    searchIconTimer = setTimeout(hideSearchIcon, 5000);
}

thinkingCheck.addEventListener('change', function() {
    if (capabilities.thinking_search_exclusive && thinkingCheck.checked) {
        searchCheck.checked = false;
    }
    sendOptions();
});

searchCheck.addEventListener('change', function() {
    if (capabilities.thinking_search_exclusive && searchCheck.checked) {
        thinkingCheck.checked = false;
    }
    sendOptions();
});

modelSelector.addEventListener('change', function() {
    const selectedModel = this.value;
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: "change_model",
            model: selectedModel
        }));
    }
});

function addThinkingMessage(text) {
    const details = document.createElement('details');
    details.className = 'message msg-thinking';
    details.open = false;

    const summary = document.createElement('summary');
    summary.textContent = '深度思考过程';
    details.appendChild(summary);

    const body = document.createElement('div');
    body.innerText = text;
    details.appendChild(body);

    chatContainer.appendChild(details);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addMessage(text, className) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${className}`;
    msgDiv.innerText = text;
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function initWebSocket() {
    ws = new WebSocket(wsUrl);

    ws.onmessage = function(event) {
        const data = JSON.parse(event.data);

        if (data.type === 'status') {
            applyStatus(data);
        }
        else if (data.type === 'config_update') {
            applyConfig(data);
        }
        else if (data.type === 'search_status') {
            if (searchCheck.checked || data.status === 'pending') {
                showSearchIcon(data.status);
            }
        }
        else if (data.type === 'thinking_msg') {
            addThinkingMessage(data.text);
        }
        else if (data.type === 'user_msg') {
            addMessage(data.text, 'msg-user');
        }
        else if (data.type === 'ai_msg') {
            addMessage(data.text, 'msg-ai');
        }
    };
}

window.addEventListener('load', initWebSocket);
